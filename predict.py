import joblib
import numpy as np
import torch
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch.optim as optim

# 🎯 Load or initialize dataset
try:
    data = joblib.load("model_data.pkl")
    if not isinstance(data, defaultdict):
        data = defaultdict(list, data)
except FileNotFoundError:
    data = defaultdict(list)

# ✅ Initialize Models
ml_model = RandomForestClassifier(n_estimators=200, random_state=42)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

aio_model = SimpleNN()
optimizer = optim.Adam(aio_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# ✅ Load GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix pad token warning

def extract_features(sequence):
    """🔍 Extract numerical features"""
    sequence = [int(x) for x in sequence if str(x).isdigit()]  # Ensure numeric values
    if not sequence:
        return [0, 0, 0]  # Prevent empty sequence errors
    return [np.mean(sequence), np.std(sequence), len(sequence)]

def train_ml_model():
    """📚 Train ML Model"""
    if not data:
        print("⚠️ No data for training!")
        return
    X, y = [], []
    for key, values in data.items():
        key = tuple(int(x) for x in key if str(x).isdigit())  # Ensure keys are numbers
        if values:
            X.extend([extract_features(list(key))] * len(values))
            y.extend(values)
    if not X or not y:
        print("⚠️ Not enough data for training!")
        return
    global ml_model
    ml_model.fit(np.array(X), np.array(y))
    print("✅ ML Model trained successfully!")

def train_ai_model():
    """🧠 Train AI Model"""
    if not data:
        print("⚠️ No data for AI training!")
        return
    X, y = [], []
    for key, values in data.items():
        key = tuple(int(x) for x in key if str(x).isdigit())  # Ensure keys are numbers
        if values:
            X.extend([extract_features(list(key))] * len(values))
            y.extend([int(v) for v in values if str(v).isdigit()])  # Ensure y contains only integers
    if not X or not y:
        print("⚠️ Not enough data for AI training!")
        return
    min_length = min(len(X), len(y))  # Ensure batch size consistency
    X, y = X[:min_length], y[:min_length]  # Trim to match lengths
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    for _ in range(50):
        optimizer.zero_grad()
        outputs = aio_model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
    print("✅ AI Model trained successfully!")

def predict_next_numbers(history):
    """🔮 Predict next numbers using ML, AI, and GPT-2"""
    if not data:
        print("⚠️ No data available for prediction!")
        return None, None, None

    history = [int(x) for x in history if str(x).isdigit()]  # Ensure numeric history
    if not history:
        print("⚠️ Invalid input: History must contain numbers!")
        return None, None, None

    features = np.array([extract_features(history)]).reshape(1, -1)

    # ✅ ML Prediction
    ml_predicted_numbers = list(map(int, ml_model.predict(features)[:4]))
    ml_probabilities = ml_model.predict_proba(features)[0]

    # ✅ AI Prediction
    ai_input = torch.tensor(features, dtype=torch.float32)
    ai_output = aio_model(ai_input).detach().numpy()[0]
    ai_predicted_numbers = list(map(int, np.argsort(ai_output)[-4:]))

    # ✅ GPT-2 Prediction
    input_text = " ".join(map(str, history)) + " ->"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = gpt2_model.generate(input_ids, max_length=len(input_ids[0]) + 1, pad_token_id=tokenizer.eos_token_id)
    gpt2_predicted_number = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("->")[-1].strip()

    try:
        gpt2_predicted_number = int(gpt2_predicted_number)
    except ValueError:
        gpt2_predicted_number = np.random.randint(0, 10)  # Fallback if GPT-2 fails

    # ✅ Merge Predictions
    final_predictions = sorted(set(ml_predicted_numbers) | set(ai_predicted_numbers) | {gpt2_predicted_number})[:4]

    # 🎯 Probability of BIG/SMALL numbers
    big_prob = sum(ml_probabilities[i] for i in range(len(ml_probabilities)) if i >= 5)
    small_prob = sum(ml_probabilities[i] for i in range(len(ml_probabilities)) if i < 5)

    return final_predictions, big_prob, small_prob

# 🎯 **Main Execution Loop**
while True:
    try:
        print("\n📝 Enter previous numbers (space-separated):")
        history = list(map(int, input().split()))
    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")
        continue

    train_ml_model()  # 🔄 Train ML
    train_ai_model()  # 🧠 Train AI

    predicted_numbers, big_prob, small_prob = predict_next_numbers(history)
    if predicted_numbers is None:
        continue

    print(f"🔮 Predicted Next Numbers: {predicted_numbers}")
    print(f"📊 Probability → BIG (>=5): {big_prob:.2%} | SMALL (<5): {small_prob:.2%}")
    
    print("🎯 Enter the actual number that appeared:")
    try:
        actual_number = int(input().strip())
        data[tuple(history)].append(actual_number)
        joblib.dump(data, "model_data.pkl")
        print("✅ Actual number saved for future training!")
    except ValueError:
        print("⚠️ Invalid input! Skipping update.")
