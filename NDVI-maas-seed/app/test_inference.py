from app.model_loader import load_latest_model
from app.inference import run_inference

def test_run_inference():
    model = load_latest_model()  # Load directly from S3 or cache
    sample_input = {
        "customerId": "test_customer",
        "loan_type": "agtech_safee",
        "source_bank": "coop",
    }
    try:
        result = run_inference(model, sample_input)
        print("Inference result:", result)
    except Exception as e:
        print("Inference failed with error:", e)

if __name__ == "__main__":
    test_run_inference()
