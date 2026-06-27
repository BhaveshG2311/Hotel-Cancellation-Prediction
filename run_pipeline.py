import os
import sys

# Add src/ folder to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from train import train_and_evaluate_models

def main():
    print("=" * 70)
    print("HOTEL CANCELLATION PREDICTION PIPELINE RUNNER")
    print("=" * 70)
    
    data_path = 'HotelData.xlsx'
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    try:
        results, best_model_name = train_and_evaluate_models(data_path)
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Best Model Selected: {best_model_name}")
        print("\nSummary of Model Performance on Test Set (20% Stratified Split):")
        
        # Display as a clean text table
        print(f"{'Algorithm':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
        print("-" * 80)
        for model_name, metrics in results.items():
            best_flag = " (Best)" if model_name == best_model_name else ""
            print(f"{model_name + best_flag:<25} | {metrics['Accuracy']:.4f}     | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}     | {metrics['F1-Score']:.4f}     | {metrics['ROC-AUC']:.4f}")
            
        print("\nGenerated Visualizations:")
        print("1. ROC Curves Comparison:   plots/roc_curve.png")
        print("2. Confusion Matrix (Best): plots/confusion_matrix.png")
        if os.path.exists('plots/feature_importance.png'):
            print("3. Feature Importance (Best): plots/feature_importance.png")
            
        print("\nSaved Artifacts:")
        print("1. Trained Best Model:      models/best_model.joblib")
        print("2. Feature Columns List:     models/feature_columns.joblib")
        
        print("\nNext Steps:")
        print("To run inference on sample bookings, execute:")
        print("  python src/predict.py --test")
        print("\nTo input custom booking details and predict, execute:")
        print("  python src/predict.py")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nPipeline execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
