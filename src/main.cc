/// std includes
#include <iostream>
#include <fstream>
#include <map>
#include <random>

/// boost includes
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/tokenizer.hpp>

/// cnn includes
#include <cnn/cnn.h>
#include <cnn/dict.h>
#include <cnn/timing.h>
#include <cnn/expr.h>
#include <cnn/nodes.h>
#include <cnn/expr.h>
#include <cnn/training.h>
#include <cnn/lstm.h>

// Local includes
#include <utils/range.h>

// For convenience bring in math operations
// on tensor expressions
using cnn::expr::operator*;
using cnn::expr::operator+;

// see handle_cli for option descriptions
struct TrainOptions {
    std::string train_file;
    std::string dev_file;
    unsigned int unk_thresh = 3;
    unsigned int input_dim = 128;
    unsigned int tag_hidden_dim = 64;
    unsigned int hidden_dim = 32;
    unsigned int layers = 3;
};

TrainOptions handle_cli(int argc, const char** argv) {
    TrainOptions opts;
    namespace po = boost::program_options;    
    po::options_description desc("\nProgram description");
    desc.add_options()
        ("help,h", "Produce this help message")
        ("train", po::value(&opts.train_file), "Sentence train file")
        ("dev", po::value(&opts.dev_file), "Sentence dev file")
        ("input_dim", po::value(&opts.input_dim), "Size of input embedding")
        ("tag_hidden_dim", po::value(&opts.tag_hidden_dim), "Size of tag embedding")
        ("layers", po::value(&opts.layers), "How many layers for the LSTM input embedding")
        ("unk-thresh", po::value(&opts.unk_thresh), "Threshold for unk");
        
    po::variables_map vm;        
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    return opts;
}

static const std::string START_SENT = "<s>";
static const std::string END_SENT = "</s>";

struct LabeledSentence {
    std::vector<int> words;
    std::vector<int> tags;
};

struct InferenceExample {
    std::vector<int> words;
    // each cnn::expr::Expression has size vocab.size()
    // and reprsents final log-scores for each output symbol
    // this can be used for inference or training
    std::vector<cnn::expr::Expression> tag_scores;
};

// A training example has an inference example
// plus expression for token log_losses
struct TrainExample {
    InferenceExample inference_example;
    std::vector<int> gold_tags;
    std::vector<cnn::expr::Expression> log_losses;

    TrainExample (InferenceExample&& ex) : inference_example(ex) {}
};

LabeledSentence read_sentence(const std::string& line, cnn::Dict& word_dict, cnn::Dict& tag_dict) {
    LabeledSentence sent;
    int idx = 0;
    sent.words.push_back(word_dict.Convert(START_SENT));
    sent.tags.push_back(tag_dict.Convert(START_SENT));
    std::istringstream buf(line);
    std::istream_iterator<std::string> beg(buf), end;
    std::vector<std::string> tokens(beg, end);
    for (const std::string& token: tokens) {
        if (idx % 2 == 0) {
            int word_id = word_dict.Convert(token);
            sent.words.push_back(word_id);
        } else {
            auto br_idx = token.find_first_of("-:+*", 1);
            std::string tag = br_idx > 0 ?
              token.substr(0, br_idx) :
              token;
            int tag_id = tag_dict.Convert(tag);
            sent.tags.push_back(tag_id);
        }
        idx++;
    }
    sent.words.push_back(word_dict.Convert(END_SENT));
    sent.tags.push_back(tag_dict.Convert(END_SENT));
    assert(sent.words.size() == sent.tags.size());
    return sent;
}

struct AffineParams {
    cnn::Parameters* W;
    cnn::Parameters* bias;
};

struct BiLSTMModel {

    cnn::Model model_;

    // Parameters
    // Input
    cnn::LookupParameters* w_;
    // project to tag hidden
    cnn::Parameters* h_l2r_;
    cnn::Parameters* h_r2l_;
    // project to tag output
    cnn::Parameters* h_bias_;
    AffineParams h2t_;

    cnn::LSTMBuilder l2r_cell_;
    cnn::LSTMBuilder r2l_cell_;
    cnn::Dict& word_dict_;
    cnn::Dict& tag_dict_;
    
    explicit BiLSTMModel(TrainOptions opts, cnn::Dict& word_dict, cnn::Dict& tag_dict) :
            model_(),
            word_dict_(word_dict),
            tag_dict_(tag_dict),
            l2r_cell_(opts.layers, opts.input_dim, opts.hidden_dim, &model_),
            r2l_cell_(opts.layers, opts.input_dim, opts.hidden_dim, &model_)
    {
        // word input
        w_ = model_.add_lookup_parameters(word_dict_.size(), {opts.input_dim});
        // word embedding to tag hidden layer
        h_l2r_ = model_.add_parameters({opts.tag_hidden_dim, opts.hidden_dim});
        h_r2l_ = model_.add_parameters({opts.tag_hidden_dim, opts.hidden_dim});
        h_bias_ = model_.add_parameters({opts.tag_hidden_dim});
        // tag embedding -> tag prediction
        h2t_.W = model_.add_parameters({tag_dict.size(), opts.tag_hidden_dim});
        h2t_.bias = model_.add_parameters({tag_dict.size()});
    }

    void reset_cell(cnn::LSTMBuilder& builder, cnn::ComputationGraph& cg, bool eval) {
        if (eval) {
            builder.disable_dropout();
        } else {
            builder.set_dropout(0.5);
        }
        builder.new_graph(cg);
        builder.start_new_sequence();
    }

    TrainExample BuildTrainExample(cnn::ComputationGraph& cg, LabeledSentence &sent, bool eval) {
        auto inference_example = this->BuildInferenceExample(cg, sent.words, eval);
        TrainExample ex{std::move(inference_example)};
        ex.gold_tags = sent.tags;
        const unsigned slen = sent.words.size();
        for (unsigned int t=0; t < slen; ++t) {
            auto tag_scores = ex.inference_example.tag_scores[t];
            // loss for current tokens
            auto tag_loss = cnn::expr::pickneglogsoftmax(tag_scores, sent.tags[t]);
            ex.log_losses.push_back(tag_loss);
        }
        return ex;
    }

    std::vector<int> Predict(std::vector<int> words) {
        cnn::ComputationGraph cg;
        auto ex = this->BuildInferenceExample(cg, words, true);
        std::vector<int> guess_tags;
        for (int t=0; t < words.size(); ++t) {
            auto tag_scores_expr = ex.tag_scores[t];
            auto tag_scores = cnn::as_vector(cg.get_value(tag_scores_expr));
            auto arg_max = std::max_element(tag_scores.begin(), tag_scores.end());
            guess_tags.push_back(std::distance(tag_scores.begin(), arg_max));
        }
        return guess_tags;
    }
    
    InferenceExample BuildInferenceExample(cnn::ComputationGraph& cg, std::vector<int> words, bool eval) {
        const unsigned slen = words.size();
        InferenceExample ex{words};
        // Tag projection
        auto W_h_l2r = cnn::expr::parameter(cg,h_l2r_);
        auto W_h_r2l = cnn::expr::parameter(cg,h_r2l_);
        auto bias_h = cnn::expr::parameter(cg,h_bias_);

        /// Tag prediction expression
        auto W_t = cnn::expr::parameter(cg, h2t_.W);
        auto bias_t = cnn::expr::parameter(cg, h2t_.bias);

        // init l2r and r2l cells
        this->reset_cell(l2r_cell_, cg, eval);
        this->reset_cell(r2l_cell_, cg, eval);

        // Inputs
        std::vector<cnn::expr::Expression> inputs;
        for (unsigned int t=0; t < slen; ++t) {
            auto& w = words[t];
            auto input = cnn::expr::lookup(cg, w_, w);
            if (!eval) {
                input = cnn::expr::noise(input, 0.1);
            }
            inputs.push_back(input);
        }

        // L2R hidden
        std::vector<cnn::expr::Expression> fwd_l2r;
        std::vector<cnn::expr::Expression> fwd_r2l;
        for (unsigned int t=0; t < slen; ++t) {
            auto l2r_idx = t;
            auto rl2_idx = slen-t-1;
            auto l2r_h = l2r_cell_.add_input(inputs[l2r_idx]);
            fwd_l2r.push_back(l2r_h);
            auto r2l_h = r2l_cell_.add_input(inputs[rl2_idx]);
            fwd_r2l.push_back(r2l_h);
        }

        // Tag output raw activations
        for (unsigned int t=0; t < slen; ++t) {
            auto lh = fwd_l2r[t];
            auto rh = fwd_r2l[t];
            auto h_raw = (W_h_l2r * lh) + (W_h_r2l * rh) + bias_h;
            auto h = cnn::expr::tanh(h_raw);
            auto t_scores = W_t * h + bias_t;
            ex.tag_scores.push_back(t_scores);
        }

        return ex;
    }    
};

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    auto opts = handle_cli(argc, const_cast<const char**>(argv));
    std::cerr << "train file is " << opts.train_file << std::endl;
    
    // Build Vocabulary
    cnn::Dict word_vocab, tag_vocab;
    auto train_lines = utils::range::istream_lines<std::ifstream>(opts.train_file);
    std::vector<LabeledSentence> train_examples;
    for (const std::string& line: train_lines) {
        auto example = read_sentence(line, word_vocab, tag_vocab);
        train_examples.push_back(example);
    }
    std::vector<LabeledSentence> dev_examples;
    auto dev_lines = utils::range::istream_lines<std::ifstream>(opts.dev_file);
    for (const std::string& line: dev_lines) {
        auto example = read_sentence(line, word_vocab, tag_vocab);
        dev_examples.push_back(example);
    }
    word_vocab.Freeze();
    tag_vocab.Freeze();
    std::cerr << "Word vocab size: " << word_vocab.size() << std::endl;
    std::cerr << "Tag vocab size: " << tag_vocab.size() << std::endl;

    // Construct Model
    BiLSTMModel model{opts, word_vocab, tag_vocab};
    cnn::AdagradTrainer sgd{&model.model_};
    unsigned int num_updates = 0; 
    for (unsigned int iter = 0; iter < 10; ++iter) {
       double loss = 0.0;
       std::cerr << "[Iteration " << iter << "]" << std::endl;
       unsigned int num_examples = 0;
       for (auto& sent: train_examples) {
           {
               cnn::ComputationGraph cg;
               auto example = model.BuildTrainExample(cg, sent, false);
               // side-effect: the last expression in the cg
               // is this sum of losses.
               auto total_loss = cnn::expr::sum(example.log_losses);
               cg.forward();
               double sent_loss = cnn::as_scalar(total_loss.value());
               loss += sent_loss;
               cg.backward();
               sgd.update(1.0);
               num_examples++;
               if (num_examples % 100 == 0) {
                   std::cerr << ".";
               }
           }
           if (num_examples % 1000 == 0) {
               std::cerr << std::endl;
               unsigned int num_corect_words = 0;
               unsigned int num_words = 0;
               cnn::Timer iter("dev eval");
               for (auto& sent: dev_examples) {
                   auto predict = model.Predict(sent.words);
                   for (unsigned int t = 1; t + 1 < sent.words.size(); ++t) {
                       int truth = sent.tags[t];
                       int guess = predict[t];
                       if (truth == guess) {
                           num_corect_words++;
                       }
                       num_words++;
                   }
               }
               double accuracy = (double) num_corect_words / (double) num_words;
               std::cerr << "Dev Accuracy: " << accuracy
               << "(" << num_corect_words << "/" << num_words << ")" << std::endl;
           }
       }
       std::cerr << "iteration loss " << loss << std::endl;
       sgd.status();
       sgd.update_epoch();
       std::shuffle(train_examples.begin(),train_examples.end(), std::default_random_engine(0L));
    }
}