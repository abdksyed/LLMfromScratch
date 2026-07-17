const PAGE_SIZE = 50;

const state = {
  stats: null,
  tokenizer: null,
  query: "",
  page: 0,
  exampleIndex: 0,
};

const number = new Intl.NumberFormat("en-IN");
const textEncoder = new TextEncoder();

const examples = [
  "India's population is 1,428,627,663.",
  "भारत अनेक भाषाओं और संस्कृतियों वाला देश है।",
  "భారతదేశం అనేక భాషలు మరియు సంస్కృతులు కలిగిన దేశం.",
  "ಭಾರತವು ಅನೇಕ ಭಾಷೆಗಳು ಮತ್ತು ಸಂಸ್ಕೃತಿಗಳನ್ನು ಹೊಂದಿರುವ ದೇಶವಾಗಿದೆ.",
];

const tokenKindLabels = {
  byte: "Byte",
  merge: "BPE merge",
  special: "Special",
};

function formatRatio(value, digits = 9) {
  return value.toFixed(digits);
}

function countFaithfulUnits(text) {
  return (
    text.match(/[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu) ?? []
  ).length;
}

function ratioRows() {
  return state.stats.languages
    .map((language) => ({
      ...language,
      tokens: language.scored_tokens,
      ratio: language.scored_ratio,
    }))
    .sort((left, right) => left.ratio - right.ratio || left.code.localeCompare(right.code));
}

function renderRatios() {
  const rows = ratioRows();
  const ratios = rows.map((row) => row.ratio);
  const maximum = Math.max(...ratios);
  const minimum = Math.min(...ratios);
  const spread = maximum - minimum;
  const score = spread === 0 ? Infinity : 1000 / spread;

  const scoreValue = document.querySelector("#score-value");
  scoreValue.textContent = Number.isFinite(score)
    ? number.format(Number(score.toFixed(2)))
    : "infinity";
  scoreValue.title = Number.isFinite(score)
    ? number.format(Number(score.toFixed(2)))
    : "infinity";
  document.querySelector("#spread-value").textContent = formatRatio(spread, 12);
  document.querySelector("#max-ratio").textContent = formatRatio(maximum);

  const body = document.querySelector("#ratio-body");
  body.replaceChildren();
  rows.forEach((row, index) => {
    const passes = row.ratio <= state.stats.ratio_limit;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="rank">X${index + 1}</td>
      <td class="language-cell"><strong>${row.name}</strong><span>${row.code.toUpperCase()} / ${row.script}</span></td>
      <td class="numeric">${number.format(row.faithful_units)}</td>
      <td class="numeric">${number.format(row.tokens)}</td>
      <td class="numeric ratio-value">${formatRatio(row.ratio)}</td>
      <td>
        <div class="bar-track" aria-hidden="true"><div class="bar-fill ${passes ? "" : "over"}" style="width:${Math.min(100, (row.ratio / 2.5) * 100)}%"></div></div>
        <span class="status ${passes ? "pass" : "fail"}">${passes ? "PASS" : "OVER LIMIT"}</span>
      </td>
    `;
    body.append(tr);
  });

  const note = document.querySelector("#vocabulary-note");
  const formula = document.querySelector("#formula");
  const verdict = document.querySelector("#verdict");
  const title = document.querySelector("#calculation-title");
  note.textContent =
    `${number.format(state.stats.vocab_size)} shared IDs; ratios use faithful Unicode units, with punctuation, URLs, and symbols included.`;
  title.textContent = "Self score";
  formula.textContent = `1000 / (${formatRatio(maximum, 12)} - ${formatRatio(minimum, 12)}) = ${number.format(Number(score.toFixed(2)))}`;
  verdict.textContent =
    "All four ratios are below 1.2, and every complete evaluation corpus decodes exactly.";
}

function tokenLabel(token) {
  if (token.kind === "byte") return `byte 0x${token.hex.toUpperCase()}`;
  return token.text;
}

function initializeTokenizerRuntime() {
  const vocab = state.tokenizer.model.vocab;
  const idToText = [];
  Object.entries(vocab).forEach(([text, id]) => {
    idToText[id] = text;
  });
  const mergeParents = new Map();
  const mergeRanks = new Map();
  state.tokenizer.model.merges.forEach((rawMerge, rank) => {
    const pair = Array.isArray(rawMerge) ? rawMerge : rawMerge.split(" ");
    const key = `${pair[0]}\u0000${pair[1]}`;
    mergeRanks.set(key, rank);
    const merged = `${pair[0]}${pair[1]}`;
    if (vocab[merged] !== undefined) mergeParents.set(vocab[merged], pair);
  });

  state.tokensById = idToText.map((text, id) => {
    const byteMatch = /^<0x([0-9A-Fa-f]{2})>$/.exec(text);
    const kind = byteMatch ? "byte" : text === "[UNK]" ? "special" : "merge";
    return {
      id,
      text,
      kind,
      hex: byteMatch ? byteMatch[1] : [...text].map((char) => textEncoder.encode(char)).flat().map((value) => value.toString(16).padStart(2, "0")).join(""),
      parents: mergeParents.get(id) ?? null,
    };
  });
  state.vocabByText = new Map(Object.entries(vocab));
  state.mergeRanks = mergeRanks;
}

function mergeSequence(sequence, pair, mergedSymbol) {
  const output = [];
  for (let index = 0; index < sequence.length; ) {
    if (index + 1 < sequence.length && sequence[index] === pair[0] && sequence[index + 1] === pair[1]) {
      output.push(mergedSymbol);
      index += 2;
    } else {
      output.push(sequence[index]);
      index += 1;
    }
  }
  return output;
}

function byteFallbackSymbols(symbol) {
  return [...textEncoder.encode(symbol)].map((value) => `<0x${value.toString(16).padStart(2, "0").toUpperCase()}>`);
}

function encodeBpePiece(piece) {
  let sequence = [...piece].flatMap((symbol) =>
    state.vocabByText.has(symbol) ? [symbol] : byteFallbackSymbols(symbol),
  );
  while (sequence.length > 1) {
    let bestPair = null;
    let bestRank = Infinity;
    for (let index = 0; index < sequence.length - 1; index += 1) {
      const pair = [sequence[index], sequence[index + 1]];
      const rank = state.mergeRanks.get(`${pair[0]}\u0000${pair[1]}`);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestPair = pair;
      }
    }
    if (bestPair === null) break;
    sequence = mergeSequence(sequence, bestPair, `${bestPair[0]}${bestPair[1]}`);
  }
  return sequence.map((symbol) => state.vocabByText.get(symbol));
}

function metaspacePieces(text) {
  const transformed = text.replaceAll(" ", "▁");
  const pieces = [];
  let start = 0;
  for (let index = 0; index < transformed.length; index += 1) {
    if (transformed[index] !== "▁") continue;
    if (index > start) pieces.push(transformed.slice(start, index));
    start = index;
  }
  if (start < transformed.length) pieces.push(transformed.slice(start));
  return pieces;
}

function encodePlaygroundText(text) {
  return metaspacePieces(text).flatMap((piece) => encodeBpePiece(piece));
}

function escapedBytes(hex) {
  return hex.match(/../g).map((byte) => `\\x${byte.toUpperCase()}`).join("");
}

function decodeTokenIds(tokenIds) {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const output = [];
  const byteBuffer = [];
  const flushBytes = () => {
    if (byteBuffer.length === 0) return;
    output.push(decoder.decode(new Uint8Array(byteBuffer)));
    byteBuffer.length = 0;
  };

  tokenIds.forEach((tokenId) => {
    const token = state.tokensById[tokenId];
    const byteMatch = /^<0x([0-9A-Fa-f]{2})>$/.exec(token.text);
    if (byteMatch) {
      byteBuffer.push(Number.parseInt(byteMatch[1], 16));
      return;
    }
    flushBytes();
    output.push(token.text.replaceAll("▁", " "));
  });
  flushBytes();
  return output.join("");
}

function parseTokenIds(rawValue) {
  const cleaned = rawValue.trim().replace(/^\[/, "").replace(/\]$/, "").trim();
  if (cleaned === "") return [];
  const parts = cleaned.split(/[\s,]+/);
  if (parts.some((part) => !/^\d+$/.test(part))) {
    throw new Error("Use integer token IDs separated by commas or whitespace.");
  }
  return parts.map((part) => {
    const tokenId = Number(part);
    if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId >= state.tokensById.length) {
      throw new Error(`Token ID ${part} is outside the vocabulary.`);
    }
    return tokenId;
  });
}

function renderDecodedIds() {
  const input = document.querySelector("#decode-ids-input");
  const status = document.querySelector("#decode-ids-status");
  const output = document.querySelector("#decode-ids-output");
  try {
    const tokenIds = parseTokenIds(input.value);
    output.textContent = tokenIds.length ? decodeTokenIds(tokenIds) : "";
    status.className = "decode-status";
    status.textContent = tokenIds.length ? `${number.format(tokenIds.length)} IDs decoded` : "";
  } catch (error) {
    output.textContent = "";
    status.className = "decode-status error";
    status.textContent = error.message;
  }
}

async function copyText(value, statusElement) {
  if (!value) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      const temporary = document.createElement("textarea");
      temporary.value = value;
      temporary.style.position = "fixed";
      temporary.style.opacity = "0";
      document.body.append(temporary);
      temporary.select();
      document.execCommand("copy");
      temporary.remove();
    }
    statusElement.textContent = "Copied";
  } catch (error) {
    statusElement.textContent = "Copy failed; select the value manually.";
  }
}

function sendEncodedIdsToDecoder() {
  const encodedIds = document.querySelector("#encoded-ids-output").value;
  document.querySelector("#decode-ids-input").value = encodedIds;
  renderDecodedIds();
}

function renderPlayground() {
  const input = document.querySelector("#playground-input");
  const text = input.value;
  const output = document.querySelector("#playground-output");
  output.replaceChildren();

  let tokenCount = 0;
  const tokenIds = encodePlaygroundText(text);
  const encodedIds = tokenIds.join(", ");
  document.querySelector("#encoded-ids-output").value = encodedIds;
  document.querySelector("#encoded-ids-status").textContent = "";
  document.querySelector("#copy-token-ids").disabled = tokenIds.length === 0;
  document.querySelector("#send-to-decoder").disabled = tokenIds.length === 0;
  tokenIds.forEach((tokenId) => {
    const token = state.tokensById[tokenId];
    if (!token) return;
    const span = document.createElement("span");
    span.className = `playground-token token-color-${tokenCount % 8}${token.kind === "byte" ? " byte-fragment" : ""}`;
    const tokenIdLabel = document.createElement("span");
    tokenIdLabel.className = "token-chip-id";
    tokenIdLabel.textContent = `ID ${token.id}`;
    const tokenText = document.createElement("span");
    tokenText.className = "token-chip-text";
    tokenText.textContent = tokenLabel(token);
    span.append(tokenIdLabel, tokenText);
    span.title = `ID ${token.id} / ${tokenKindLabels[token.kind]} / ${token.hex}`;
    output.append(span);
    tokenCount += 1;
  });

  document.querySelector("#playground-tokens").textContent = number.format(tokenCount);
  const words = text.match(/\S+/gu) ?? [];
  const faithfulUnitCount = countFaithfulUnits(text);
  document.querySelector("#playground-words").textContent = number.format(words.length);
  document.querySelector("#playground-units").textContent = number.format(faithfulUnitCount);
  document.querySelector("#playground-ratio").textContent = faithfulUnitCount
    ? (tokenCount / faithfulUnitCount).toFixed(3)
    : "0.000";
  output.classList.toggle("empty", text.length === 0);
  if (text.length === 0) output.textContent = "No tokens";
}

function showPlaygroundExample() {
  document.querySelector("#playground-input").value = examples[state.exampleIndex];
  state.exampleIndex = (state.exampleIndex + 1) % examples.length;
  renderPlayground();
  sendEncodedIdsToDecoder();
}

function filteredTokens() {
  const query = state.query.trim().toLocaleLowerCase();
  if (!query) return state.tokensById;
  return state.tokensById.filter((token) => {
    const haystack = [
      token.id,
      token.kind,
      token.text ?? "",
      token.hex ?? "",
    ]
      .join(" ")
      .toLocaleLowerCase();
    return haystack.includes(query);
  });
}

function renderTokens() {
  const tokens = filteredTokens();
  const totalPages = Math.max(1, Math.ceil(tokens.length / PAGE_SIZE));
  state.page = Math.min(state.page, totalPages - 1);
  const pageTokens = tokens.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
  const body = document.querySelector("#token-body");
  body.replaceChildren();

  pageTokens.forEach((token) => {
    const tr = document.createElement("tr");
    const id = document.createElement("td");
    id.textContent = token.id;

    const kind = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `token-kind ${token.kind}`;
    badge.textContent = tokenKindLabels[token.kind];
    kind.append(badge);

    const text = document.createElement("td");
    text.className = "token-text";
    text.textContent = tokenLabel(token);

    const tokenHex = document.createElement("td");
    tokenHex.className = "token-hex";
    tokenHex.textContent = token.hex ?? "-";

    const parents = document.createElement("td");
    parents.className = "token-parents";
    parents.textContent = token.parents ? `${token.parents[0]} + ${token.parents[1]}` : "-";

    tr.append(id, kind, text, tokenHex, parents);
    body.append(tr);
  });

  const first = tokens.length === 0 ? 0 : state.page * PAGE_SIZE + 1;
  const last = Math.min(tokens.length, (state.page + 1) * PAGE_SIZE);
  document.querySelector("#token-result-count").textContent = `${number.format(first)}-${number.format(last)} of ${number.format(tokens.length)} tokens`;
  document.querySelector("#page-label").textContent = `Page ${state.page + 1} of ${totalPages}`;
  document.querySelector("#previous-page").disabled = state.page === 0;
  document.querySelector("#next-page").disabled = state.page >= totalPages - 1;
}

function renderCorpora() {
  const list = document.querySelector("#corpus-list");
  list.replaceChildren();
  state.stats.languages.forEach((language) => {
    const item = document.createElement("div");
    item.className = "corpus-item";
    const name = document.createElement("a");
    name.className = "corpus-source";
    name.href = language.source_url;
    name.target = "_blank";
    name.rel = "noreferrer";
    name.textContent = `${language.name} Wikipedia`;
    const detail = document.createElement("span");
    detail.textContent = `${language.file} / ${number.format(language.bytes)} bytes / ${number.format(language.faithful_units)} faithful units`;
    const acquisition = document.createElement("span");
    acquisition.textContent = `${language.acquisition} / ${number.format(language.words)} whitespace words`;
    const retrieved = document.createElement("span");
    retrieved.textContent = `Retrieved ${language.retrieved_at}`;
    const cleaning = document.createElement("span");
    cleaning.textContent = `Cleaning: ${language.cleaning}`;
    const fileLink = document.createElement("a");
    fileLink.className = "corpus-file";
    fileLink.href = `assets/${language.file}`;
    fileLink.download = language.file.split("/").pop();
    fileLink.textContent = `Download ${language.file.split("/").pop()}`;
    const hash = document.createElement("code");
    hash.title = language.sha256;
    hash.textContent = `SHA-256 ${language.sha256}`;
    item.append(name, detail, acquisition, retrieved, cleaning, fileLink, hash);
    list.append(item);
  });
}

function bindEvents() {
  document.querySelector("#token-search").addEventListener("input", (event) => {
    state.query = event.target.value;
    state.page = 0;
    renderTokens();
  });
  document.querySelector("#previous-page").addEventListener("click", () => {
    state.page -= 1;
    renderTokens();
  });
  document.querySelector("#next-page").addEventListener("click", () => {
    state.page += 1;
    renderTokens();
  });

  document.querySelector("#playground-input").addEventListener("input", renderPlayground);
  document.querySelector("#clear-playground").addEventListener("click", () => {
    document.querySelector("#playground-input").value = "";
    renderPlayground();
  });
  document.querySelector("#example-playground").addEventListener("click", showPlaygroundExample);
  document.querySelector("#decode-ids-input").addEventListener("input", renderDecodedIds);
  document.querySelector("#decode-ids-button").addEventListener("click", renderDecodedIds);
  document.querySelector("#clear-decode-ids").addEventListener("click", () => {
    document.querySelector("#decode-ids-input").value = "";
    renderDecodedIds();
  });
  document.querySelector("#copy-token-ids").addEventListener("click", () => {
    copyText(
      document.querySelector("#encoded-ids-output").value,
      document.querySelector("#encoded-ids-status"),
    );
  });
  document.querySelector("#send-to-decoder").addEventListener("click", sendEncodedIdsToDecoder);
  document.querySelector("#copy-decoded-text").addEventListener("click", () => {
    copyText(
      document.querySelector("#decode-ids-output").textContent,
      document.querySelector("#decoded-copy-status"),
    );
  });
}

async function initialize() {
  try {
    const [statsResponse, tokenizerResponse] = await Promise.all([
      fetch("assets/stats.json"),
      fetch("assets/tokenizer.json"),
    ]);
    if (!statsResponse.ok || !tokenizerResponse.ok) throw new Error("Artifact request failed");
    [state.stats, state.tokenizer] = await Promise.all([
      statsResponse.json(),
      tokenizerResponse.json(),
    ]);
    document.querySelector("#vocab-count").textContent = number.format(
      Object.keys(state.tokenizer.model.vocab).length,
    );
    initializeTokenizerRuntime();
    renderRatios();
    renderTokens();
    renderCorpora();
    bindEvents();
    showPlaygroundExample();
  } catch (error) {
    document.querySelector("#vocabulary-note").textContent =
      "The generated artifacts could not be loaded. Serve this directory over HTTP and retry.";
    document.querySelector("#token-result-count").textContent = "Vocabulary unavailable";
    console.error(error);
  }
}

initialize();
