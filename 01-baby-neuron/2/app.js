import { generateRings } from "../shared/js/data.js";
import { drawPlot } from "../shared/js/draw.js";
import {
  accuracy,
  makeLinearModel,
  predictLinear,
  trainLinearEpoch
} from "../shared/js/ring-models.js";
import {
  collapseDeepLinear,
  makeDeepLinearModel,
  makeDeepReluModel,
  predictDeepLinear,
  predictDeepRelu,
  trainDeepLinearEpoch,
  trainDeepReluEpoch,
  formatDeepLinearMatrices,
  traceDeepLinearCalculation
} from "../shared/js/deep-models.js";
import { RandomSource } from "../shared/js/random.js";
import { $, bindEpochSlider, epochsPerFrame, formatPercent, nextFrame } from "../shared/js/ui.js";
import { formatFixed } from "../shared/js/math.js";

const els = {
  epochs: $("#epochs"),
  epochLabel: $("#epochLabel"),
  rerun: $("#rerun"),
  train: $("#train"),
  linearCanvas: $("#linearCanvas"),
  deepLinearCanvas: $("#deepLinearCanvas"),
  deepReluCanvas: $("#deepReluCanvas"),
  linearAccuracy: $("#linearAccuracy"),
  deepLinearAccuracy: $("#deepLinearAccuracy"),
  deepReluAccuracy: $("#deepReluAccuracy"),
  collapseText: $("#collapseText"),
  collapseMatrix: $("#collapseMatrix"),
  matrixDetails: $("#matrixDetails"),
  status: $("#status")
};

const random = new RandomSource(19);

let points = [];
let linearModel = null;
let deepLinearModel = null;
let deepReluModel = null;
let trainingToken = 0;

function resetModels() {
  linearModel = makeLinearModel(random);
  deepLinearModel = makeDeepLinearModel(random);
  deepReluModel = makeDeepReluModel(random);
}

function refreshCollapseText() {
  const collapsed = collapseDeepLinear(deepLinearModel);
  const [w0, w1] = collapsed.weights.map((value) => formatFixed(value));
  const b = formatFixed(collapsed.bias);
  els.collapseText.textContent = `The five weight matrices multiply down to one 1x2 matrix plus one bias.`;
  els.collapseMatrix.textContent = [
    "Shapes: (1x4)(4x4)(4x4)(4x4)(4x2) -> (1x2)",
    `Product W = [${w0}, ${w1}]`,
    `Bias b    = ${b}`,
    `So: sigmoid(${w0} * x + ${w1} * y + ${b})`
  ].join("\n");

  const proofPoint = points[0] || { x: 0.75, y: -0.35 };
  els.matrixDetails.textContent = [
    traceDeepLinearCalculation(deepLinearModel, [proofPoint.x, proofPoint.y]),
    "",
    "The actual learned matrices:",
    formatDeepLinearMatrices(deepLinearModel)
  ].join("\n");
}

function refreshView() {
  const linearPredict = (x, y) => predictLinear(linearModel, x, y);
  const deepLinearPredict = (x, y) => predictDeepLinear(deepLinearModel, x, y);
  const deepReluPredict = (x, y) => predictDeepRelu(deepReluModel, x, y);

  els.linearAccuracy.textContent = formatPercent(accuracy(points, linearPredict));
  els.deepLinearAccuracy.textContent = formatPercent(accuracy(points, deepLinearPredict));
  els.deepReluAccuracy.textContent = formatPercent(accuracy(points, deepReluPredict));

  drawPlot(els.linearCanvas, points, linearPredict);
  drawPlot(els.deepLinearCanvas, points, deepLinearPredict);
  drawPlot(els.deepReluCanvas, points, deepReluPredict);
  refreshCollapseText();
}

function regenerateData() {
  random.bump();
  points = generateRings(random);
  resetModels();
  refreshView();
  els.status.textContent = `Generated ${points.length} new points. Models are untrained.`;
}

async function trainModels() {
  const token = ++trainingToken;
  const epochs = Number(els.epochs.value);
  const frameBatch = epochsPerFrame(epochs);
  let completedEpochs = 0;

  els.status.textContent = "Training...";
  els.train.disabled = true;

  while (completedEpochs < epochs && token === trainingToken) {
    for (let step = 0; step < frameBatch && completedEpochs < epochs; step++) {
      trainLinearEpoch(linearModel, points);
      trainDeepLinearEpoch(deepLinearModel, points);
      trainDeepReluEpoch(deepReluModel, points);
      completedEpochs++;
    }

    refreshView();
    els.status.textContent = `Training epoch ${completedEpochs} of ${epochs}...`;
    await nextFrame();
  }

  if (token === trainingToken) {
    els.status.textContent = `Trained ${points.length} points for ${epochs} epochs.`;
    els.train.disabled = false;
  }
}

bindEpochSlider(els.epochs, els.epochLabel);

els.rerun.addEventListener("click", () => {
  trainingToken++;
  els.train.disabled = false;
  regenerateData();
});

els.train.addEventListener("click", () => {
  trainingToken++;
  resetModels();
  refreshView();
  trainModels();
});

resetModels();
regenerateData();
