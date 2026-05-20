import random

def gradient_descent_health_game():
    # 1. Scenario: Healthcare Predictive Analytics
    print("--- 🩺 THE GRADIENT GLIDER: HEALTH MONITOR 🩺 ---")
    print("Mission: Predict Blood Pressure based on a patient's Heart Rate.")
    print("Goal: Use Gradient Descent to optimize your line of best fit.")

    # 2. Generating Synthetic Patient Records (Heart Rate, Blood Pressure)
    # The true underlying relationship here is: BP = 1.2 * HR + 10
    patients = [
        {"hr": 60, "bp": 82},
        {"hr": 80, "bp": 106},
        {"hr": 100, "bp": 130}
    ]
    
    print("\n--- 🖥️ HOSPITAL RECORD TRIALS (TRAINING DATA) ---")
    for idx, p in enumerate(patients):
        print(f"Patient {idx+1}: Heart Rate = {p['hr']} bpm | Blood Pressure = {p['bp']} mmHg")

    # 3. Game Inputs: Initial Weight (Slope) and Bias (Intercept)
    print("\n--- STEP 1: INITIALIZE YOUR MODEL ---")
    print("We need to map a line: Prediction = (Heart Rate * Weight) + Bias")
    try:
        weight = float(input("Enter starting Weight / Slope (e.g., 0.5): "))
        bias = float(input("Enter starting Bias / Intercept (e.g., 5.0): "))
        epochs = int(input("Enter Training Iterations / Epochs (e.g., 3): "))
    except ValueError:
        weight, bias, epochs = 0.5, 5.0, 3

    # 4. The Optimization Loop: Gradient Descent
    # We use a tiny learning rate so the math doesn't explode
    learning_rate = 0.0001 
    
    print(f"\n--- 🔄 COMMENCING OVERSIGHT TRAINING FOR {epochs} EPOCHS ---")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        weight_gradient = 0
        bias_gradient = 0
        n = len(patients)
        
        for p in patients:
            x = p["hr"]
            y = p["bp"]
            
            # Step A: Generate continuous prediction
            prediction = (x * weight) + bias
            
            # Step B: Calculate Mean Squared Error Loss (Error Squared)
            error = prediction - y
            total_loss += error ** 2
            
            # Step C: Compute Gradients (Partial Derivatives)
            weight_gradient += (2/n) * error * x
            bias_gradient += (2/n) * error
            
        mse_loss = total_loss / n
        
        # Step D: Update Parameters along the negative gradient
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        
        print(f"Epoch {epoch}: Loss (MSE) = {mse_loss:.2f} | Weight = {weight:.4f} | Bias = {bias:.4f}")

    # 5. Diagnostic Evaluation
    print("\n--- 📊 DIAGNOSTIC RESULTS ---")
    print(f"Final Optimized Model: BP = (HR * {weight:.2f}) + {bias:.2f}")
    
    if mse_loss < 50:
        print("🏆 SUCCESS: Outstanding optimization! Your health monitor safely minimizes risk.")
    elif mse_loss < 500:
        print("✅ STABLE: Fairly accurate alignment, but could benefit from more epochs.")
    else:
        print("⚠️ CRITICAL ERROR: High loss! High prediction errors in healthcare are dangerous.")

if __name__ == "__main__":
    gradient_descent_health_game()
