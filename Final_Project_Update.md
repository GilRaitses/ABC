
# Final Project Update

Here's a detailed outline on how to approach each "To Do" item in the project update, incorporating best practices and tools in R:

## 1. Data Preparation:

- **Import:** Ensure your data set contains raw accelerometer data for each axis (x, y, z).
- **Filtering/Smoothing:** Apply a smoothing filter (e.g., moving average) to reduce noise if needed.
- **Calibration (Optional):** If your accelerometer data needs calibration to account for sensor bias or drift, perform this step first.

## 2. Calculate Roll and Pitch:

- **Libraries:** Use the `madgwickfilter` or `RSpincalc` R package, which provide functions to estimate roll and pitch angles from accelerometer and optionally gyroscope data.
- **Coordinate System:** Ensure you understand the coordinate system used by your accelerometer (e.g., right-hand rule) to interpret the angle calculations correctly.
- **Algorithm:** Apply a suitable algorithm (e.g., Madgwick filter) to compute the roll and pitch angles for each time step.
- **Validation:** Visually inspect the calculated angles against the raw accelerometer data or video footage (if available) to ensure they make sense and align with the observed whale movements.

## 3. Feature Extraction:

- **Roll Rate and Pitch Rate:** Calculate the rate of change (derivative) of the roll and pitch angles over time to capture the dynamic aspects of these features.
- **Additional Features (Optional):** Consider calculating features like the variance or standard deviation of roll and pitch angles within a sliding window to capture the variability of these movements.

### Code Example

\`\`\`r
library(madgwickfilter) # or library(RSpincalc) - Choose one
library(dplyr)

# Assuming your data is in a dataframe called 'whale_data'
# with columns 'acceleration_x', 'acceleration_y', 'acceleration_z', and 'timestamp'

# 1. Calculate Roll and Pitch:
whale_data <- whale_data %>%
  mutate(
    # Use madgwickfilter::MadgwickAHRS or RSpincalc::eulerAngles - Choose one
    angles = MadgwickAHRS(
      SampleRate = 1 / mean(diff(timestamp)), # Calculate average sampling rate
      Beta = 0.1, # Tuning parameter - adjust as needed
      Acc = cbind(acceleration_x, acceleration_y, acceleration_z)
    )
  ) %>%
  tidyr::unnest_wider(angles) %>%  # Expand the 'angles' list into columns
  rename(roll = roll, pitch = pitch) # Rename columns if needed

# 2. Calculate Roll Rate and Pitch Rate:
whale_data <- whale_data %>%
  mutate(
    roll_rate  = c(0, diff(roll))  / c(diff(timestamp), 1),
    pitch_rate = c(0, diff(pitch)) / c(diff(timestamp), 1)
  )
\`\`\`

### Explanation

- \`MadgwickAHRS/eulerAngles\`: These functions estimate roll and pitch from accelerometer data. Adjust the \`SampleRate\` and \`Beta\` parameters if necessary.
- \`tidyr::unnest_wider\`: Expands the resulting 'angles' list-column into separate columns.

## 4. Data Preparation:

- **Feature Scaling:** Standardize or normalize your features to ensure they have similar scales. This can improve the performance of some machine learning algorithms.
- **Split Data:** Divide your labeled dataset into training and testing sets. A common split is 80% for training and 20% for testing.

## 5. Model Training:

- **Library:** Use the \`randomForest\` package in R.
- **Model Parameters:** Tune hyperparameters like \`ntree\` (number of trees) and \`mtry\` (number of variables randomly sampled at each split) to optimize model performance.
- **Cross-Validation:** Use k-fold cross-validation to estimate the model's generalization performance and prevent overfitting.

## 6. Model Evaluation:

- **Metrics:** Assess model performance using metrics like accuracy, precision, recall, and F1-score.
- **Confusion Matrix:** Examine the confusion matrix to understand the types of errors the model makes.

### Code Example

\`\`\`r
library(randomForest)
library(caret)

# Assuming 'foraging_event' is your binary target variable (0/1)

# 1. Split Data:
set.seed(123) # For reproducibility
train_index <- createDataPartition(whale_data$foraging_event, p = 0.8, list = FALSE)
train_data <- whale_data[train_index, ]
test_data  <- whale_data[-train_index, ]

# 2. Train Model:
rf_model <- randomForest(foraging_event ~ norm_jerk + roll_rate + pitch_rate + depth,
                         data = train_data, ntree = 500, mtry = 3) # Adjust 'ntree' and 'mtry' as needed

# 3. Predict and Evaluate:
predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(predictions, test_data$foraging_event)
\`\`\`

### Explanation

- \`createDataPartition\`: Splits data into training and testing sets.
- \`randomForest\`: Trains a random forest model using specified features.
- \`predict\`: Generates predictions on the test set.
- \`confusionMatrix\`: Evaluates model performance.

## 7. Data Preparation:

- Focus on the norm-jerk signal and potentially other selected features.

## 8. Threshold Selection:

- **Manual:** Start with an initial guess based on domain knowledge or visual inspection of the data.
- **Statistical:** Consider using methods like percentiles or standard deviations from the mean to set thresholds.
- **Adaptive:** If the signal characteristics vary, explore adaptive thresholding techniques.

## 9. Event Detection:

- Identify points where the feature(s) exceed the threshold.
- Consider additional criteria like minimum peak prominence or duration to filter out false positives.

## 10. Evaluation:

- Compare the detected events to your labeled data to assess performance.
- Calculate metrics like precision, recall, and F1-score to quantify accuracy.

### Code Example

\`\`\`r
# Example using norm_jerk (you can try other features too)
threshold <- 0.5 # Adjust this based on your data exploration
whale_data <- whale_data %>%
  mutate(foraging_event_threshold = ifelse(norm_jerk > threshold, 1, 0))

# Evaluate
confusionMatrix(whale_data$foraging_event_threshold, whale_data$foraging_event)
\`\`\`

### Explanation

- Sets a threshold value for the \`norm_jerk\` signal.
- Creates a new variable (\`foraging_event_threshold\`) indicating events based on this threshold.
- Evaluates against the ground truth \`foraging_event\` to see how well the threshold works.

Let me know if you'd like a more detailed code example or further guidance on any of these steps!
