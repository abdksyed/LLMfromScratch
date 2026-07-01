import { RandomSource } from "../shared/js/random.js";
import { dot } from "../shared/js/math.js";
import { $, bindEpochSlider, epochsPerFrame, nextFrame } from "../shared/js/ui.js";

const animals = ["cat", "dog", "cow"];
const fruits = ["apple", "mango"];
const verbs = ["eat", "chase", "see"];
const supportTokens = ["the", ".", "tastes", "sweet"];
const tokens = [...animals, ...fruits, ...verbs, ...supportTokens];
const tokenToIndex = new Map(tokens.map((token, index) => [token, index]));

const categoryByToken = new Map([
  ...animals.map((token) => [token, "animal"]),
  ...fruits.map((token) => [token, "fruit"]),
  ...verbs.map((token) => [token, "verb"])
]);

const categoryColor = {
  animal: "#246bfe",
  fruit: "#e34f4f",
  verb: "#1c8b5a",
  other: "#8a8175"
};

const els = {
  epochs: $("#epochs"),
  epochLabel: $("#epochLabel"),
  reset: $("#reset"),
  train: $("#train"),
  status: $("#status"),
  lossValue: $("#lossValue"),
  embeddingCanvas: $("#embeddingCanvas"),
  neighborsBody: $("#neighborsBody"),
  grammarText: $("#grammarText")
};

const embeddingSize = 8;
const random = new RandomSource(31);
const sentences = buildSentences();
const pairs = buildPairs(sentences);

let model = null;
let trainingToken = 0;

function buildSentences() {
  const result = [];

  for (const animal of animals) {
    for (const verb of verbs) {
      for (const otherAnimal of animals) {
        result.push(["the", animal, verb, "the", otherAnimal, "."]);
      }
    }

    for (const fruit of fruits) {
      result.push(["the", animal, "eat", "the", fruit, "."]);
    }
  }

  for (const fruit of fruits) {
    result.push(["the", fruit, "tastes", "sweet", "."]);
  }

  return result;
}

function buildPairs(sentenceList) {
  const result = [];

  for (const sentence of sentenceList) {
    for (let i = 0; i < sentence.length - 1; i++) {
      result.push([tokenToIndex.get(sentence[i]), tokenToIndex.get(sentence[i + 1])]);
    }
  }

  return result;
}

function makeModel() {
  return {
    embeddings: tokens.map(() => Array.from({ length: embeddingSize }, () => random.normal() * 0.18)),
    outputWeights: Array.from({ length: embeddingSize }, () =>
      Array.from({ length: tokens.length }, () => random.normal() * 0.18)
    ),
    outputBias: Array(tokens.length).fill(0)
  };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function predictNext(modelToUse, inputIndex) {
  const embedding = modelToUse.embeddings[inputIndex];
  const logits = modelToUse.outputBias.map((bias, tokenIndex) => {
    let sum = bias;
    for (let dim = 0; dim < embeddingSize; dim++) {
      sum += embedding[dim] * modelToUse.outputWeights[dim][tokenIndex];
    }
    return sum;
  });

  return softmax(logits);
}

function trainEpoch() {
  const rate = 0.08;
  let totalLoss = 0;

  for (const [inputIndex, targetIndex] of pairs) {
    const embedding = model.embeddings[inputIndex];
    const probabilities = predictNext(model, inputIndex);
    totalLoss += -Math.log(Math.max(probabilities[targetIndex], 1e-9));

    const outputError = probabilities.slice();
    outputError[targetIndex] -= 1;

    const embeddingGradient = Array(embeddingSize).fill(0);
    for (let dim = 0; dim < embeddingSize; dim++) {
      for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
        embeddingGradient[dim] += model.outputWeights[dim][tokenIndex] * outputError[tokenIndex];
      }
    }

    for (let dim = 0; dim < embeddingSize; dim++) {
      for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
        model.outputWeights[dim][tokenIndex] -= rate * outputError[tokenIndex] * embedding[dim];
      }
    }

    for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
      model.outputBias[tokenIndex] -= rate * outputError[tokenIndex];
    }

    for (let dim = 0; dim < embeddingSize; dim++) {
      embedding[dim] -= rate * embeddingGradient[dim];
    }
  }

  return totalLoss / pairs.length;
}

function projectEmbeddings() {
  const contentTokens = [...animals, ...fruits, ...verbs];
  const vectors = contentTokens.map((token) => model.embeddings[tokenToIndex.get(token)]);
  const mean = Array(embeddingSize).fill(0);

  for (const vector of vectors) {
    for (let dim = 0; dim < embeddingSize; dim++) mean[dim] += vector[dim] / vectors.length;
  }

  const centered = vectors.map((vector) => vector.map((value, dim) => value - mean[dim]));
  const pc1 = powerDirection(centered);
  const pc2 = powerDirection(centered.map((vector) => subtractProjection(vector, pc1)));
  const points = contentTokens.map((token, index) => ({
    token,
    category: categoryByToken.get(token),
    x: dot(centered[index], pc1),
    y: dot(centered[index], pc2)
  }));

  return normalizePoints(points);
}

function powerDirection(vectors) {
  let direction = Array.from({ length: embeddingSize }, (_, index) => (index === 0 ? 1 : 0));

  for (let step = 0; step < 30; step++) {
    const next = Array(embeddingSize).fill(0);
    for (const vector of vectors) {
      const scale = dot(vector, direction);
      for (let dim = 0; dim < embeddingSize; dim++) next[dim] += scale * vector[dim];
    }
    direction = normalize(next);
  }

  return direction;
}

function subtractProjection(vector, direction) {
  const scale = dot(vector, direction);
  return vector.map((value, index) => value - scale * direction[index]);
}

function normalize(vector) {
  const length = Math.hypot(...vector) || 1;
  return vector.map((value) => value / length);
}

function normalizePoints(points) {
  const maxAbs = Math.max(...points.flatMap((point) => [Math.abs(point.x), Math.abs(point.y)]), 1e-6);
  return points.map((point) => ({
    ...point,
    x: point.x / maxAbs,
    y: point.y / maxAbs
  }));
}

function drawEmbeddings() {
  const ctx = els.embeddingCanvas.getContext("2d");
  const { width, height } = els.embeddingCanvas;
  const points2d = projectEmbeddings();

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfaf7";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(23, 32, 38, 0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(width / 2, 0);
  ctx.lineTo(width / 2, height);
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  for (const point of points2d) {
    const px = width / 2 + point.x * width * 0.38;
    const py = height / 2 - point.y * height * 0.38;

    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.fillStyle = categoryColor[point.category];
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#172026";
    ctx.font = "700 15px ui-sans-serif, system-ui";
    ctx.fillText(point.token, px + 13, py + 5);
  }
}

function cosineSimilarity(left, right) {
  return dot(left, right) / ((Math.hypot(...left) || 1) * (Math.hypot(...right) || 1));
}

function renderNeighbors() {
  const contentTokens = [...animals, ...fruits, ...verbs];
  els.neighborsBody.replaceChildren();

  for (const token of contentTokens) {
    const vector = model.embeddings[tokenToIndex.get(token)];
    const nearest = contentTokens
      .filter((candidate) => candidate !== token)
      .map((candidate) => ({
        token: candidate,
        category: categoryByToken.get(candidate),
        score: cosineSimilarity(vector, model.embeddings[tokenToIndex.get(candidate)])
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 1);

    const row = document.createElement("tr");
    row.innerHTML = `
      <td><strong>${token}</strong></td>
      <td>${categoryByToken.get(token)}</td>
      <td>${nearest.map((item) => `${item.token} (${item.score.toFixed(2)})`).join(", ")}</td>
    `;
    els.neighborsBody.append(row);
  }
}

function refresh(loss = null) {
  els.lossValue.textContent = loss === null ? "--" : loss.toFixed(2);
  drawEmbeddings();
  renderNeighbors();
}

function resetDemo() {
  trainingToken++;
  random.bump();
  model = makeModel();
  els.train.disabled = false;
  els.status.textContent = "Fresh random embeddings. Train to reveal clusters.";
  refresh();
}

async function trainModel() {
  const token = ++trainingToken;
  const epochs = Number(els.epochs.value);
  const frameBatch = epochsPerFrame(epochs);
  let completed = 0;
  let loss = 0;

  els.train.disabled = true;
  els.status.textContent = "Training next-token model...";

  while (completed < epochs && token === trainingToken) {
    for (let step = 0; step < frameBatch && completed < epochs; step++) {
      loss = trainEpoch();
      completed++;
    }

    refresh(loss);
    els.status.textContent = `Training epoch ${completed} of ${epochs}...`;
    await nextFrame();
  }

  if (token === trainingToken) {
    els.status.textContent = "Trained only on next-token prediction. Similar tokens now cluster.";
    els.train.disabled = false;
  }
}

function renderGrammar() {
  els.grammarText.textContent = [
    "Templates:",
    "the {animal} {verb} the {animal} .",
    "the {animal} eat the {fruit} .",
    "the {fruit} tastes sweet .",
    "",
    `Animals: ${animals.join(", ")}`,
    `Fruits:  ${fruits.join(", ")}`,
    `Verbs:   ${verbs.join(", ")}`,
    "",
    `Training pairs: ${pairs.length}`,
    "Loss is next-token cross entropy."
  ].join("\n");
}

bindEpochSlider(els.epochs, els.epochLabel);

els.reset.addEventListener("click", resetDemo);
els.train.addEventListener("click", trainModel);

renderGrammar();
resetDemo();
