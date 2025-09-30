import requests
import json

# Test data
test_data = {
    "Time_spent_Alone": 4,
    "Stage_fear": "No",
    "Social_event_attendance": 4,
    "Going_outside": 6,
    "Drained_after_socializing": "No",
    "Friends_circle_size": 13,
    "Post_frequency": 5
}

print("Testing the API...")

try:
    # Send request to your local API
    response = requests.post("http://localhost:5000/predict", json=test_data)
    
    # Check if successful
    if response.status_code == 200:
        result = response.json()
        print("✅ SUCCESS! API is working!")
        print(f"Prediction: {result['prediction']}")
        print("Probabilities:")
        for personality, prob in result['probabilities'].items():
            print(f"  {personality}: {prob:.2%}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())
        
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    print("Make sure you ran: python app.py")