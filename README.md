
# SMS Spam Detection

This project identifies spam SMS messages using a machine learning model. The project includes data cleaning, preprocessing, and model training steps.

## Dataset

The dataset contains SMS messages labeled as "ham" (non-spam) or "spam." It is loaded from a CSV file named `spam.csv`.

## Project Structure

1. **Data Loading**: Loads the dataset and displays sample data points.
2. **Data Cleaning**: Handles missing values and irrelevant columns.
3. **Data Preprocessing**: Includes tokenization, removing stop words, and converting text data to numerical format.
4. **Model Training**: A machine learning model is trained to classify messages as spam or ham.
5. **Evaluation**: The model is evaluated on a test set to measure accuracy and performance metrics.

## Dependencies

- Python 3.11
- NumPy
- Pandas
- Scikit-learn

Install dependencies with:

```bash
pip install numpy pandas scikit-learn
```

## Usage

1. Clone the repository.
2. Run the notebook `SMSspam.ipynb` to execute the data cleaning, preprocessing, model training, and evaluation.

## Results

The model's accuracy and other evaluation metrics are displayed at the end of the notebook.

## License

This project is licensed under the MIT License.
