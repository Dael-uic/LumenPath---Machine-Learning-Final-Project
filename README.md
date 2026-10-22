# LumenPath Learning — Website Project

A responsive, offline-friendly static website for the ELEC026.23 Machine Learning final project. LumenPath Learning is a fictional education-technology company that demonstrates six core machine learning algorithms in an education-sector scenario.

## Open the website

1. Extract the project ZIP (if you downloaded the ZIP).
2. Keep `index.html`, `styles.css`, `script.js`, and `student-data.js` together in the same folder. Keep the CSV in the `data` folder.
3. Double-click `index.html` to open it in a modern browser. No package installation or internet connection is required.

For a local development server, open a terminal in this folder and run:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Files

- `index.html` — page structure, company information, algorithm explanations, data explorer, interactive lab, and team section.
- `styles.css` — responsive layout and visual theme.
- `script.js` — dataset explorer and browser-side demonstrations for all six algorithms.
- `student-data.js` — 36 generated fictional learner records used by the interactive models.
- `data/student_learning_sample.csv` — the same sample dataset as a CSV download.

## Machine learning demonstrations

- **Linear Regression:** multiple linear regression fit to estimate a numeric final score from four indicators.
- **Naive Bayes:** categorical Naive Bayes classifier with Laplace smoothing to estimate a support-flag likelihood.
- **K-Nearest Neighbors:** compares a profile with the three closest sample records using scaled features.
- **Support Vector Machine:** a simplified linear SVM trained with batch sub-gradient descent on hinge loss.
- **Decision Tree:** a shallow CART-style classification tree with Gini-based splits.
- **Artificial Neural Network:** a small feed-forward network with one hidden layer, trained by backpropagation in the browser.

The models run locally in the browser after the page loads; no form data is sent to a server. The model implementations are intentionally small and educational, not production-grade ML libraries.

## Dataset and responsible-use note

The records are fabricated. Each sample's `at_risk` label is generated using a transparent demonstration rule: final score below 65, attendance below 65%, or both prior score below 55 and assignment average below 65. These labels are not teacher judgements, diagnoses, or evidence about real students. The sample is small, unvalidated, and not suitable for educational decisions. A real deployment would require appropriate consent and governance, privacy protections, representative data, bias and performance evaluation, and human review.

## Team

- Amper
- Ariego
- Dael
- Malaga

## Project objective coverage

The site includes a fictional company overview, mission, vision, goals, education-related data need, all six required algorithms, data justification and interpretations, a searchable/filterable sample dataset with CSV download, interactive model outputs, and a responsible-use discussion.
