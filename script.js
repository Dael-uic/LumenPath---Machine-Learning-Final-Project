(() => {
  'use strict';

  const students = Array.isArray(window.STUDENT_DATA) ? window.STUDENT_DATA : [];
  const featureDefs = [
    { key: 'attendance_pct', label: 'Attendance', scale: 100 },
    { key: 'practice_hours', label: 'Practice hours', scale: 10 },
    { key: 'prior_score', label: 'Prior score', scale: 100 },
    { key: 'assignment_avg', label: 'Assignment average', scale: 100 }
  ];
  const RISK_THRESHOLD = 0.5;

  const $ = (id) => document.getElementById(id);
  const controls = {
    attendance: $('attendanceInput'),
    practice: $('practiceInput'),
    prior: $('priorInput'),
    assignment: $('assignmentInput'),
    model: $('modelSelect')
  };

  const sigmoid = (value) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const mean = (values) => values.reduce((sum, v) => sum + v, 0) / (values.length || 1);
  const isRisk = (student) => Number(student.at_risk) === 1;
  const featureVector = (profile) => [
    Number(profile.attendance_pct) / 100,
    Number(profile.practice_hours) / 10,
    Number(profile.prior_score) / 100,
    Number(profile.assignment_avg) / 100
  ];

  function currentProfile() {
    return {
      attendance_pct: Number(controls.attendance.value),
      practice_hours: Number(controls.practice.value),
      prior_score: Number(controls.prior.value),
      assignment_avg: Number(controls.assignment.value)
    };
  }

  // Multiple linear regression, solved with the normal equation and a small ridge term
  // to keep the matrix stable for a compact demonstration dataset.
  function solveLinearSystem(matrix, vector) {
    const n = vector.length;
    const a = matrix.map((row, i) => [...row, vector[i]]);
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      for (let row = col + 1; row < n; row += 1) {
        if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
      }
      [a[col], a[pivot]] = [a[pivot], a[col]];
      if (Math.abs(a[col][col]) < 1e-10) continue;
      const divisor = a[col][col];
      for (let j = col; j <= n; j += 1) a[col][j] /= divisor;
      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = a[row][col];
        for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
      }
    }
    return a.map((row) => Number.isFinite(row[n]) ? row[n] : 0);
  }

  function fitLinearRegression() {
    const p = students.map((s) => [1, ...featureVector(s)]);
    const y = students.map((s) => Number(s.final_score));
    const n = p[0].length;
    const xtx = Array.from({ length: n }, () => Array(n).fill(0));
    const xty = Array(n).fill(0);
    for (let r = 0; r < p.length; r += 1) {
      for (let i = 0; i < n; i += 1) {
        xty[i] += p[r][i] * y[r];
        for (let j = 0; j < n; j += 1) xtx[i][j] += p[r][i] * p[r][j];
      }
    }
    for (let i = 1; i < n; i += 1) xtx[i][i] += 1e-5;
    return solveLinearSystem(xtx, xty);
  }

  const regressionWeights = students.length ? fitLinearRegression() : [0, 0, 0, 0, 0];
  function predictRegression(profile) {
    const x = [1, ...featureVector(profile)];
    const score = regressionWeights.reduce((sum, weight, i) => sum + weight * x[i], 0);
    return clamp(score, 0, 100);
  }

  function attendanceBand(value) {
    return value < 75 ? 'low' : value < 90 ? 'medium' : 'high';
  }
  function practiceBand(value) {
    return value < 4 ? 'low' : value < 7 ? 'medium' : 'high';
  }
  function scoreBand(value) {
    return value < 60 ? 'low' : value < 80 ? 'medium' : 'high';
  }
  function categoricalFeatures(profile) {
    return [
      attendanceBand(profile.attendance_pct),
      practiceBand(profile.practice_hours),
      scoreBand(profile.prior_score),
      scoreBand(profile.assignment_avg)
    ];
  }

  // Categorical Naive Bayes with Laplace smoothing.
  function predictNaiveBayes(profile) {
    const categories = students.map(categoricalFeatures);
    const query = categoricalFeatures(profile);
    const priors = [0, 1].map((label) => students.filter((s) => Number(s.at_risk) === label).length);
    const logProb = [0, 1].map((label) => {
      const classRows = students.filter((s) => Number(s.at_risk) === label);
      let total = Math.log((priors[label] + 1) / (students.length + 2));
      for (let featureIndex = 0; featureIndex < query.length; featureIndex += 1) {
        const matching = classRows.reduce((count, row) => {
          return count + (categoricalFeatures(row)[featureIndex] === query[featureIndex] ? 1 : 0);
        }, 0);
        total += Math.log((matching + 1) / (classRows.length + 3));
      }
      return total;
    });
    const probability = sigmoid(logProb[1] - logProb[0]);
    return { probability, risk: probability >= RISK_THRESHOLD, bands: query };
  }

  function predictKnn(profile, k = 3) {
    const target = featureVector(profile);
    const ranked = students.map((student) => {
      const vector = featureVector(student);
      const distance = Math.sqrt(vector.reduce((sum, value, index) => sum + (value - target[index]) ** 2, 0));
      return { student, distance };
    }).sort((a, b) => a.distance - b.distance);
    const neighbors = ranked.slice(0, Math.min(k, ranked.length));
    const riskVotes = neighbors.filter((entry) => isRisk(entry.student)).length;
    return { neighbors, riskVotes, count: neighbors.length, probability: neighbors.length ? riskVotes / neighbors.length : 0 };
  }

  // Linear SVM trained with batch sub-gradient descent on hinge loss. Standardizing
  // the input dimensions helps the margin compare indicators with different units.
  const svmTrainingVectors = students.map(featureVector);
  const svmMeans = featureDefs.map((_, i) => mean(svmTrainingVectors.map((vector) => vector[i])));
  const svmScales = featureDefs.map((_, i) => {
    const variance = mean(svmTrainingVectors.map((vector) => (vector[i] - svmMeans[i]) ** 2));
    return Math.sqrt(variance) || 1;
  });
  function svmVector(profile) {
    return featureVector(profile).map((value, i) => (value - svmMeans[i]) / svmScales[i]);
  }
  function fitSvm() {
    const examples = students.map((student) => ({ x: svmVector(student), y: isRisk(student) ? 1 : -1 }));
    let weights = [0, 0, 0, 0];
    let bias = 0;
    const lambda = 0.02;
    const learningRate = 0.035;
    for (let epoch = 0; epoch < 1400; epoch += 1) {
      const gradient = weights.map((weight) => lambda * weight);
      let biasGradient = 0;
      for (const example of examples) {
        const margin = example.y * (weights.reduce((sum, weight, i) => sum + weight * example.x[i], 0) + bias);
        if (margin < 1) {
          for (let i = 0; i < weights.length; i += 1) gradient[i] -= (example.y * example.x[i]) / examples.length;
          biasGradient -= example.y / examples.length;
        }
      }
      weights = weights.map((weight, i) => weight - learningRate * gradient[i]);
      bias -= learningRate * biasGradient;
    }
    return { weights, bias };
  }
  const svmModel = students.length ? fitSvm() : { weights: [0, 0, 0, 0], bias: 0 };
  function predictSvm(profile) {
    const x = svmVector(profile);
    const margin = svmModel.weights.reduce((sum, weight, i) => sum + weight * x[i], svmModel.bias);
    const probabilityProxy = sigmoid(margin * 2.4);
    return { margin, probability: probabilityProxy, risk: margin >= 0 };
  }

  // A small CART-style classification tree selected by Gini impurity.
  function gini(rows) {
    if (!rows.length) return 0;
    const riskRate = rows.filter(isRisk).length / rows.length;
    return 1 - riskRate * riskRate - (1 - riskRate) * (1 - riskRate);
  }
  function majorityRisk(rows) {
    return rows.filter(isRisk).length >= rows.length / 2;
  }
  function trainTree(rows, depth = 0, maxDepth = 3) {
    const positives = rows.filter(isRisk).length;
    if (!rows.length || depth >= maxDepth || positives === 0 || positives === rows.length || rows.length < 5) {
      return { leaf: true, risk: rows.length ? majorityRisk(rows) : false, count: rows.length };
    }
    const baseImpurity = gini(rows);
    let best = null;
    for (const feature of featureDefs) {
      const unique = [...new Set(rows.map((row) => Number(row[feature.key])))].sort((a, b) => a - b);
      const thresholds = unique.slice(1).map((value, i) => (value + unique[i]) / 2);
      for (const threshold of thresholds) {
        const left = rows.filter((row) => Number(row[feature.key]) <= threshold);
        const right = rows.filter((row) => Number(row[feature.key]) > threshold);
        if (left.length < 3 || right.length < 3) continue;
        const weighted = (left.length * gini(left) + right.length * gini(right)) / rows.length;
        const gain = baseImpurity - weighted;
        if (!best || gain > best.gain) best = { feature, threshold, left, right, gain };
      }
    }
    if (!best || best.gain < 0.005) return { leaf: true, risk: majorityRisk(rows), count: rows.length };
    return {
      leaf: false,
      feature: best.feature,
      threshold: best.threshold,
      left: trainTree(best.left, depth + 1, maxDepth),
      right: trainTree(best.right, depth + 1, maxDepth),
      count: rows.length
    };
  }
  const decisionTree = students.length ? trainTree(students) : { leaf: true, risk: false, count: 0 };
  function predictTree(profile) {
    let node = decisionTree;
    const path = [];
    while (node && !node.leaf) {
      const value = Number(profile[node.feature.key]);
      const goesLeft = value <= node.threshold;
      const featureLabel = node.feature.key === 'practice_hours' ? 'Practice' : node.feature.label;
      const thresholdText = node.feature.key === 'practice_hours' ? `${node.threshold.toFixed(1)} hrs` : `${Math.round(node.threshold)}%`;
      path.push(`${featureLabel} ${goesLeft ? '≤' : '>'} ${thresholdText}`);
      node = goesLeft ? node.left : node.right;
    }
    const risk = node ? node.risk : false;
    path.push(risk ? 'support flag' : 'no support flag');
    return { risk, path, leafCount: node ? node.count : 0 };
  }

  // Tiny one-hidden-layer neural network trained by backpropagation on the sample rows.
  function makeRng(seed) {
    let value = seed >>> 0;
    return () => {
      value = (1664525 * value + 1013904223) >>> 0;
      return value / 4294967296;
    };
  }
  function fitAnn() {
    const rng = makeRng(20261022);
    const inputs = students.map((student) => featureVector(student));
    const labels = students.map((student) => isRisk(student) ? 1 : 0);
    const hiddenCount = 6;
    const w1 = Array.from({ length: 4 }, () => Array.from({ length: hiddenCount }, () => (rng() - 0.5) * 0.8));
    const b1 = Array.from({ length: hiddenCount }, () => (rng() - 0.5) * 0.2);
    const w2 = Array.from({ length: hiddenCount }, () => (rng() - 0.5) * 0.8);
    let b2 = 0;
    const order = inputs.map((_, i) => i);
    const learningRate = 0.22;
    for (let epoch = 0; epoch < 700; epoch += 1) {
      // Deterministic order is sufficient for a reproducible teaching demo.
      for (const rowIndex of order) {
        const x = inputs[rowIndex];
        const y = labels[rowIndex];
        const hidden = b1.map((bias, j) => sigmoid(bias + x.reduce((sum, value, i) => sum + value * w1[i][j], 0)));
        const output = sigmoid(b2 + hidden.reduce((sum, value, j) => sum + value * w2[j], 0));
        const outputDelta = output - y;
        const hiddenDeltas = hidden.map((value, j) => outputDelta * w2[j] * value * (1 - value));
        for (let j = 0; j < hiddenCount; j += 1) w2[j] -= learningRate * outputDelta * hidden[j];
        b2 -= learningRate * outputDelta;
        for (let i = 0; i < 4; i += 1) {
          for (let j = 0; j < hiddenCount; j += 1) w1[i][j] -= learningRate * hiddenDeltas[j] * x[i];
        }
        for (let j = 0; j < hiddenCount; j += 1) b1[j] -= learningRate * hiddenDeltas[j];
      }
    }
    return { w1, b1, w2, b2 };
  }
  const annModel = students.length ? fitAnn() : null;
  function predictAnn(profile) {
    if (!annModel) return { probability: 0.5, risk: false };
    const x = featureVector(profile);
    const hidden = annModel.b1.map((bias, j) => sigmoid(bias + x.reduce((sum, value, i) => sum + value * annModel.w1[i][j], 0)));
    const probability = sigmoid(annModel.b2 + hidden.reduce((sum, value, j) => sum + value * annModel.w2[j], 0));
    return { probability, risk: probability >= RISK_THRESHOLD };
  }

  function setMetric(main, unit = '') {
    const metric = $('resultMetric');
    metric.textContent = String(main);
    if (unit) {
      const small = document.createElement('small');
      small.textContent = unit;
      metric.appendChild(small);
    }
    metric.classList.toggle('word-metric', String(main).length > 7);
  }

  function updateResult({ title, label, metric, unit = '', bar = 0, summary, interpretation, data, caution }) {
    $('resultTitle').textContent = title;
    $('resultMetricLabel').textContent = label;
    setMetric(metric, unit);
    $('resultTrackFill').style.width = `${clamp(Number(bar) || 0, 0, 100)}%`;
    $('resultSummary').textContent = summary;
    $('resultInterpretation').textContent = interpretation;
    $('resultData').textContent = data;
    $('resultCaution').textContent = caution || 'Predictions are demonstrations, not recommendations about a real learner.';
  }

  function runModel() {
    if (!students.length) {
      updateResult({ title: 'Dataset unavailable', label: 'MODEL OUTPUT', metric: '—', summary: 'The sample dataset could not be loaded.', interpretation: 'Open index.html together with student-data.js in the same project folder.', data: 'No records loaded.', caution: 'Keep all provided project files in their original folder structure.' });
      return;
    }
    const profile = currentProfile();
    const selected = controls.model.value;
    const dataBlurb = `${students.length} synthetic records · attendance, practice, prior score, and assignment average.`;
    const generalCaution = 'This is a classroom demonstration trained on fabricated data and rule-generated labels. It is not validated for use with real students.';

    if (selected === 'regression') {
      const score = Math.round(predictRegression(profile));
      updateResult({
        title: 'Linear Regression', label: 'ESTIMATED FINAL SCORE', metric: score, unit: '/100', bar: score,
        summary: `The model estimates a final score of ${score} out of 100 for this profile.`,
        interpretation: 'The model fits a linear relationship between the four indicators and the sample final scores. This is a numeric estimate, not a support classification.',
        data: `${students.length} records · final score is the target; four learning indicators are the inputs.`, caution: 'The estimate is affected by the small synthetic sample and should not be treated as a promise of future performance.'
      });
      return;
    }
    if (selected === 'naivebayes') {
      const result = predictNaiveBayes(profile);
      const probability = Math.round(result.probability * 100);
      updateResult({
        title: 'Naive Bayes', label: 'ESTIMATED SUPPORT-FLAG PROBABILITY', metric: `${probability}%`, bar: probability,
        summary: result.risk ? 'The model leans toward a support review for this profile.' : 'The model does not assign the stronger support flag to this profile.',
        interpretation: `The model combined the likelihood of these bands: attendance ${result.bands[0]}, practice ${result.bands[1]}, prior score ${result.bands[2]}, and assignments ${result.bands[3]}. Naive Bayes treats each feature as conditionally independent within each class.`,
        data: dataBlurb, caution: generalCaution
      });
      return;
    }
    if (selected === 'knn') {
      const result = predictKnn(profile, 3);
      const percent = Math.round(result.probability * 100);
      const safeVotes = result.count - result.riskVotes;
      updateResult({
        title: 'K-Nearest Neighbors', label: 'NEAREST-NEIGHBOR VOTE', metric: `${result.riskVotes} / ${result.count}`, unit: 'flagged', bar: percent,
        summary: `${result.riskVotes} of the ${result.count} most similar sample profiles have a support flag; ${safeVotes} do not.`,
        interpretation: 'KNN finds the three closest profiles using scaled values for attendance, practice time, prior score, and assignment average. The majority label becomes the demonstration output.',
        data: `${students.length} records · K = ${result.count} neighbors · input features are scaled to comparable ranges.`, caution: generalCaution
      });
      return;
    }
    if (selected === 'svm') {
      const result = predictSvm(profile);
      updateResult({
        title: 'Support Vector Machine', label: 'PREDICTED CLASS', metric: result.risk ? 'Review' : 'On track', unit: 'sample class', bar: result.risk ? 100 : 0,
        summary: result.risk ? 'The profile falls on the support-flag side of the learned linear boundary.' : 'The profile falls on the no-flag side of the learned linear boundary.',
        interpretation: `The signed decision margin is ${result.margin.toFixed(3)} (${result.margin >= 0 ? 'positive' : 'negative'} in this demo's convention). Distance is a model score, not a calibrated probability.`,
        data: `${students.length} labelled records · linear decision boundary · four scaled indicators.`, caution: 'This simplified SVM is trained on fabricated labels. Its margin is not a probability, and the boundary may change with new data.'
      });
      return;
    }
    if (selected === 'tree') {
      const result = predictTree(profile);
      const probability = result.risk ? 100 : 0;
      updateResult({
        title: 'Decision Tree', label: 'PREDICTED CLASS', metric: result.risk ? 'Review' : 'On track', unit: 'sample class', bar: probability,
        summary: result.risk ? 'The decision path ends at a leaf with a support flag.' : 'The decision path ends at a leaf without a support flag.',
        interpretation: `Learned path: ${result.path.join(' → ')}. A decision tree makes its prediction by following feature thresholds from the root to a leaf. The displayed path is the route this profile took.`,
        data: `${students.length} labelled records · CART-style Gini splits · maximum depth of 3.`, caution: 'A short path is easier to explain, but it may hide uncertainty and reflect quirks in the small sample.'
      });
      return;
    }
    if (selected === 'ann') {
      const result = predictAnn(profile);
      const probability = Math.round(result.probability * 100);
      updateResult({
        title: 'Artificial Neural Network', label: 'ESTIMATED SUPPORT-FLAG PROBABILITY', metric: `${probability}%`, bar: probability,
        summary: result.risk ? 'The network output is above the demo threshold for a support flag.' : 'The network output is below the demo threshold for a support flag.',
        interpretation: 'A small neural network combines four normalized inputs through six hidden units and a sigmoid output. Backpropagation adjusts its weights to reduce prediction error on the generated training examples.',
        data: `${students.length} labelled records · 4 inputs · 6 hidden units · 1 sigmoid output.`, caution: 'Neural networks can learn complex patterns but are harder to interpret. This score is not calibrated or validated for real-world use.'
      });
    }
  }

  function updateRangeValues() {
    $('attendanceValue').textContent = `${controls.attendance.value}%`;
    $('practiceValue').textContent = `${Number(controls.practice.value).toFixed(1)} hrs`;
    $('priorValue').textContent = `${controls.prior.value} / 100`;
    $('assignmentValue').textContent = `${controls.assignment.value} / 100`;
  }

  let showAll = false;
  function renderDataset() {
    const searchText = ($('datasetSearch').value || '').trim().toLowerCase();
    const filter = $('riskFilter').value;
    const filtered = students.filter((student) => {
      const matchesSearch = String(student.student_id).toLowerCase().includes(searchText);
      const risk = isRisk(student);
      const matchesFilter = filter === 'all' || (filter === 'risk' && risk) || (filter === 'ontrack' && !risk);
      return matchesSearch && matchesFilter;
    });
    const shown = showAll ? filtered : filtered.slice(0, 8);
    const body = $('datasetBody');
    body.replaceChildren();
    for (const student of shown) {
      const row = document.createElement('tr');
      const cells = [
        { value: student.student_id },
        { value: `${student.attendance_pct}%` },
        { value: `${Number(student.practice_hours).toFixed(1)} hrs` },
        { value: student.prior_score },
        { value: `${student.assignment_avg}%` },
        { value: student.final_score, className: 'score-pill' },
        { value: isRisk(student) ? 'Support flag' : 'No flag', className: isRisk(student) ? 'risk-tag' : 'risk-tag safe' }
      ];
      cells.forEach((cell, index) => {
        const td = document.createElement('td');
        if (cell.className) {
          const badge = document.createElement('span');
          badge.className = cell.className;
          badge.textContent = cell.value;
          td.appendChild(badge);
        } else {
          td.textContent = cell.value;
        }
        if (index === 0) td.setAttribute('scope', 'row');
        row.appendChild(td);
      });
      body.appendChild(row);
    }
    $('datasetRange').textContent = filtered.length
      ? `Showing ${shown.length} of ${filtered.length} matching records`
      : 'No matching records';
    $('showAllRows').hidden = filtered.length <= 8;
    $('showAllRows').textContent = showAll ? 'Show first 8 rows →' : 'Show all rows →';
  }

  function init() {
    if ($('heroStudentCount')) $('heroStudentCount').textContent = String(students.length || 0);
    if ($('datasetCount')) $('datasetCount').textContent = String(students.length || 0);
    if ($('riskCount')) $('riskCount').textContent = String(students.filter(isRisk).length);
    updateRangeValues();
    renderDataset();
    runModel();

    Object.values(controls).filter((control) => control && control.type === 'range').forEach((control) => {
      control.addEventListener('input', updateRangeValues);
    });
    $('runModel').addEventListener('click', runModel);
    $('resetDemo').addEventListener('click', () => {
      controls.attendance.value = 86;
      controls.practice.value = 6;
      controls.prior.value = 76;
      controls.assignment.value = 80;
      controls.model.value = 'regression';
      updateRangeValues();
      runModel();
    });
    controls.model.addEventListener('change', runModel);
    $('datasetSearch').addEventListener('input', () => { showAll = false; renderDataset(); });
    $('riskFilter').addEventListener('change', () => { showAll = false; renderDataset(); });
    $('showAllRows').addEventListener('click', () => { showAll = !showAll; renderDataset(); });

    document.querySelectorAll('[data-model]').forEach((button) => {
      button.addEventListener('click', () => {
        controls.model.value = button.dataset.model;
        runModel();
        $('lab').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      });
    });

    const menuToggle = $('menuToggle');
    const nav = $('primaryNav');
    menuToggle.addEventListener('click', () => {
      const open = menuToggle.getAttribute('aria-expanded') !== 'true';
      menuToggle.setAttribute('aria-expanded', String(open));
      menuToggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
      nav.classList.toggle('is-open', open);
    });
    nav.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
      nav.classList.remove('is-open');
      menuToggle.setAttribute('aria-expanded', 'false');
      menuToggle.setAttribute('aria-label', 'Open navigation');
    }));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
