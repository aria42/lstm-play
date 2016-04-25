/// std includes
#include <iostream>
#include <fstream>
#include <map>

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
#include "utils/src/Utils.h"

struct TrainOptions {
    std::string train_file;
    std::string dev_file;
    unsigned int unk_thresh = 3;
    unsigned int input_dim = 256;
    unsigned int hidden_dim = 128;
    unsigned int layers = 1;
};

TrainOptions handle_cli(int argc, char** argv) {
    TrainOptions opts;
    namespace po = boost::program_options;    
    po::options_description desc("\nProgram description");
    desc.add_options()
        ("help,h", "Produce this help message")
        ("train", po::value(&opts.train_file), "Sentence train file")
        ("dev", po::value(&opts.dev_file), "Sentence dev file")
        ("unk-thresh", po::value(&opts.unk_thresh), "Threshold for unk");
        
    po::variables_map vm;        
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    return opts;
}

void each_line(const std::string& path, std::function<void(const std::string&)> fn) {
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        fn(line);
    }
}

std::vector<int> read_sentence(const std::string& line, cnn::Dict& dict) {
    std::vector<int> ret;
    int start_id = dict.Convert("<s>");
    ret.push_back(start_id);    
    for (const char& ch: line) {
        int id = dict.Convert("" + ch);
        ret.push_back(id);
    }    
    int stop_id = dict.Convert("</s>");
    ret.push_back(stop_id);
    return ret;
}

struct Example {
    std::vector<int> sent;
    cnn::ComputationGraph cg;
    // each cnn::expr::Expression has size vocab.size()
    // and reprsents final log-scores for each output symbol
    std::vector<cnn::expr::Expression> input_scores;
};

struct CharLM {    
    // Params
    cnn::LookupParameters* char_lookup_;
    cnn::Parameters* W_param_;
    cnn::Parameters* bias_param_;
        
    cnn::LSTMBuilder cell_builder_;
    cnn::Model model_;    
    cnn::Dict& vocab_;
    
    explicit CharLM(TrainOptions opts, cnn::Dict& vocab) : 
            vocab_(vocab),
            cell_builder_(opts.layers, opts.input_dim, opts.hidden_dim, &model_)
    {
        char_lookup_ = model_.add_lookup_parameters(vocab.size(), {opts.input_dim});        
        W_param_ = model_.add_parameters({vocab.size(), opts.hidden_dim});
        bias_param_ = model_.add_parameters({vocab.size()});
    }
    
    std::vector<const std::string*> Predict(const std::vector<int>& sent) {
        auto example = this->BuildSentenceLogLoss(sent, true);
        example.cg.forward();
        std::vector<const std::string*> outputs;
        for (unsigned t=0; t < sent.size(); ++t) {
            auto& activations = example.input_scores[t];
            auto scores = cnn::as_vector(example.cg.get_value(activations));
            auto arg_max = std::max_element(scores.begin(), scores.end());
            int idx = std::distance(scores.begin(), arg_max);                        
            outputs.push_back(&vocab_.Convert(idx)); 
        } 
        return outputs;
    } 
    
    Example BuildSentenceLogLoss(const std::vector<int>& sent, bool eval = false) {
        const unsigned slen = sent.size();
        Example example{sent};
        cnn::ComputationGraph& cg = example.cg;
        
        if (!eval) {
            cell_builder_.set_dropout(0.5);
        } else {
            cell_builder_.disable_dropout();
        }
                
        cell_builder_.new_graph(cg);        
        /// Variables for each time-step
        auto W = cnn::expr::parameter(cg, W_param_);
        auto bias = cnn::expr::parameter(cg, bias_param_);
        /// Losses
        std::vector<cnn::expr::Expression> char_losses;
         
        cell_builder_.start_new_sequence();        
        for (unsigned int t=0; t < sent.size(); ++t) {
            auto& x = sent[t];
            auto input = cnn::expr::lookup(cg, char_lookup_, x);
            if (!eval) {
                input = cnn::expr::noise(input, 1.0);
            }
            // final hidden state for cell
            auto hidden = cell_builder_.add_input(input);          
            auto activation = W * hidden + bias;
            auto prediction = cnn::expr::pickneglogsoftmax(activation, x);
            example.input_scores.push_back(activation);
            char_losses.push_back(prediction);                         
        }
        // side-effect, adds the sum of losses 
        // as the final node in the cg
        cnn::expr::sum(char_losses);               
        
        return example;
    }    
};

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    auto opts = handle_cli(argc, argv);           
    std::cerr << "train file is " << opts.train_file << std::endl;
    
    // Build Vocabulary
    cnn::Dict vocab;       
    each_line(opts.train_file, [&vocab](const auto& line) {
        read_sentence(line, vocab);
    });
    each_line(opts.dev_file, [&vocab](const auto& line) {
        read_sentence(line, vocab);
    });
    vocab.Freeze();
    std::cerr << "Vocab size: " << vocab.size() << std::endl;
    
    // Construct Model
    CharLM model{opts, vocab};
    cnn::AdagradTrainer sgd{&model.model_};
    unsigned int num_updates = 0; 
    for (unsigned int iter = 0; iter < 10; ++iter) {
       double loss = 0.0;
       std::cerr << "[Iteration " << iter << "]" << std::endl;
       unsigned int num_examples = 0; 
       each_line(opts.train_file, [&](const auto& line) {          
          {
            auto sent = read_sentence(line, vocab);
            auto example = model.BuildSentenceLogLoss(sent, false);
            double sent_loss = cnn::as_scalar(example.cg.forward());
            loss += sent_loss;
            example.cg.backward();
            sgd.update(1.0);           
            num_examples++;                
          }
          if (num_examples % 1000 == 0) {
            unsigned int num_corect_words = 0;
            unsigned int num_words = 0;
            cnn::Timer iter("iteration took");       
            each_line(opts.dev_file, [&](const auto& line) {              
              auto sent = read_sentence(line, vocab);
              auto predict = model.Predict(sent);
              for (unsigned int t=0; t < sent.size(); ++t) {
                auto& truth = sent[t];
                auto predict_ptr = predict[t];
                if (truth == vocab.Convert(*predict_ptr)) {
                  num_corect_words++;  
                } 
                num_words++;                
              }                
            });
            double accuracy = (double)num_corect_words/(double)num_words;
            std::cerr << "Dev Accuracy: " << accuracy 
                      << "(" << num_corect_words << "/" << num_words << ")" << std::endl;            
          }                                      
       });
       std::cerr << "iteration loss " << loss << std::endl;
       sgd.status();
       sgd.update_epoch(); 
    }
}