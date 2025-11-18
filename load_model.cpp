#include <torch/script.h> 
#include <torch/torch.h>
#include <iostream>
#include <chrono>
int main(const int argc, const char* argv[]) {
    // try {
        // Load the scripted model
        torch::jit::script::Module module = torch::jit::load(argv[1]);
        std::cout << "Model loaded successfully\n";
        double total_loss = 0.0;

        auto dataset_module = torch::jit::load("scripted_test_data.pt");
        auto t0 = std::chrono::high_resolution_clock::now();
        // auto num_graphs = dataset_module.attr("num_graphs").toTensor();
        // std::cout << "Number of graphs in dataset: " << num_graphs << "\n";
        int num_graphs =3784;
        for (int i=0; i<num_graphs; i++){
            // std::cout << "Processing graph " << i << "\n";
            auto x = dataset_module.attr(("x_"+std::to_string(i)).c_str()).toTensor();
            auto edge_index = dataset_module.attr(("edge_index_"+std::to_string(i)).c_str()).toTensor();
            auto edge_attr = dataset_module.attr(("edge_attr_"+std::to_string(i)).c_str()).toTensor();
            auto y = dataset_module.attr(("y_"+std::to_string(i)).c_str()).toTensor();
            // std::cout << "Processing graph " << edge_index<< "\n";
            torch::Tensor batch = torch::zeros({42}, torch::dtype(torch::kLong));

            std::vector<torch::jit::IValue> inputs = { edge_index, x,edge_attr, batch};
            auto output = module.forward(inputs).toTensor();
            auto loss = torch::nn::functional::mse_loss(output, y);
            
            // std::cout << "Loss for graph " << i << ": " << loss.item<double>() << "\n";
            
            // std::cout << "Output " << i << ": " << output.item<double>() << "\n";
            // std::cout << "Actual " << i << ": " << y.item<double>() << "\n";
            total_loss= total_loss+ loss.item<double>();

                     
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double mean_loss = total_loss / num_graphs;
        std::cout << "Mean Loss over " << num_graphs << " graphs: " << mean_loss << "\n";
        std::cout<<"took"<<std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()<<std::endl;

    // catch (const c10::Error& e) {
    //     std::cerr << "Error loading the model\n";
    //     return -1;
    // }
}
