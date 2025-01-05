![alt text](iitkgplogo.jpg)


# Adaptive Graph Pooling for Protein Structure Classification Using GCNs  

## **Objective**  
The objective of this project is to implement and evaluate a novel graph pooling algorithm combining adaptive node down-sampling (gPool) and hierarchical pooling (diffPool) for graph classification tasks. The model is tested on benchmark protein structure datasets (D&D and ENZYMES) using Graph Convolutional Networks (GCNs) and optimized with various hyperparameter configurations.  

## **Model Architecture**  
The implemented model architecture follows this sequence:  
`GNN1 → GNN2 → Down-Sample & Pool1 → GNN3 → GNN4 → Down-Sample & Pool2 → Classification Head`  

- **Down-Sampling Layer (gPool):** Selects the most important k%-nodes adaptively based on scalar projection values on a trainable projection vector.  
- **Hierarchical Pooling (diffPool):** Forms a coarsened graph with m-clusters of nodes and learns modified node features and adjacency matrices.  

## **Tasks**  
1. **Experimentation with GCN:**  
   - Evaluate the model using GCN layers [Kipf et al.] and report comparative results.  
2. **Down-Sampling (gPool):**  
   - Experiment with k = {90%, 80%, 60%} to identify the optimal percentage of important nodes to retain.  
3. **Hierarchical Pooling (diffPool):**  
   - Experiment with m = 6 and m = 3 for Down-Sample & Pool1 and Down-Sample & Pool2, respectively.  
4. **Dataset Evaluation:**  
   - Use the D&D dataset for binary classification and ENZYMES dataset for 6-class classification.  
   - Divide each dataset into Training (80%), Validation (10%), and Testing (10%) splits.  
   - Report classification accuracy for the test set after optimizing performance on the validation set.  

## **Tech Stacks**  
- **Programming Language:** Python  
- **Graph Neural Network Frameworks:** PyTorch Geometric (PyG) or DGL  
- **Deep Learning Framework:** PyTorch  
- **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn  
- **Evaluation Tools:** Scikit-learn for metrics and hyperparameter tuning  
- **Environment:** Jupyter Notebook or Google Colab  

## **How to Run the Project**  
1. Clone the repository:  
   ```bash  
   git clone <repository-url>  
   cd <repository-name>  
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the training script:  
   ```bash  
   python train_graph_pooling.py  
   ```  

## **Results**  
- **Down-Sample & Pool1 (k = 90%, m = 6):** Results reported on both datasets.  
- **Down-Sample & Pool2 (k = 60%, m = 3):** Results reported on both datasets.  
- Comprehensive comparison of GCN performance with different hyperparameters.  

Feel free to explore and experiment with the model by adjusting hyperparameters or datasets!  
