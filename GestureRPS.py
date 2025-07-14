import cv2
import numpy as np

# Function to classify gestures based on convexity defects
def classify_gesture(contour):
    if contour is None or cv2.contourArea(contour) < 5000:
        return None

    # Calculate the convex hull and defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return None

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return None

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate the angle between start, far, and end points
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

        # Count fingers if the angle is less than 90 degrees and defect depth is significant
        if angle <= np.pi / 2 and d > 10000:
            finger_count += 1

    # Determine gesture based on the number of fingers detected
    if finger_count == 0:
        return 'rock'
    elif finger_count == 1:  # One space between two fingers indicates scissors
        return 'scissors'
    elif finger_count >= 3:  # Multiple fingers extended for paper
        return 'paper'
    return None

# Deterministic function for computer's choice
def computer_choice(counter):
    options = ["rock", "paper", "scissors"]
    return options[counter % 3]  # Cycle through options


# Initialize variables
user_points = 0
computer_points = 0
counter = 0  # Counter for computer's deterministic choice

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    print("\nShow your gesture! (Press 'c' to capture or 'q' to quit)")

    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not working")
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Display the camera feed
        cv2.putText(frame, "Press 'c' to capture or 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Rock Paper Scissors", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture gesture
            break
        elif key == ord('q'):  # Quiq# t game
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Convert to HSV and create a skin color mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological transformations to clean up the mask
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    user_input = None
    if contours:
        # Find the largest contour (assuming it's the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Classify the gesture
        user_input = classify_gesture(max_contour)

    if user_input:
        # Determine computer's input
        computer_input = computer_choice(counter)
        counter += 1  # Increment counter for deterministic choice

        # Game logic
        if user_input == computer_input:
            result = "It's a tie!"
        elif (user_input == "rock" and computer_input == "scissors") or \
                (user_input == "paper" and computer_input == "rock") or \
                (user_input == "scissors" and computer_input == "paper"):
            result = "You win!"
            user_points += 1
        else:
            result = "Computer wins!"
            computer_points += 1

        # Display the results
        print(f"\nUser input: {user_input}, Computer input: {computer_input}")
        print(result)
        print(f"Score: User: {user_points}, Computer: {computer_points}")
    else:
        print("Couldn't detect a valid gesture. Try again!")
cap.release()
cv2.destroyAllWindows()