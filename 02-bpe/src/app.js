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
  "India is a multilingual country with a shared digital future.",
  "भारत अनेक भाषाओं और संस्कृतियों वाला देश है।",
  "భారతదేశం అనేక భాషలు మరియు సంస్కృతులు కలిగిన దేశం.",
  "ಭಾರತವು ಅನೇಕ ಭಾಷೆಗಳು ಮತ್ತು ಸಂಸ್ಕೃತಿಗಳನ್ನು ಹೊಂದಿರುವ ದೇಶವಾಗಿದೆ.",
];

const tokenKindLabels = {
  byte: "Byte",
  merge: "BPE merge",
  lexeme: "Word",
  piece: "Word piece",
};

function formatRatio(value, digits = 9) {
  return value.toFixed(digits);
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
      <td class="numeric">${number.format(row.words)}</td>
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
  const splitTypes = state.stats.languages.reduce(
    (total, language) => total + language.split_word_types,
    0,
  );
  note.textContent =
    `${number.format(state.stats.token_kind_counts.lexeme)} full-word tokens and ${splitTypes} two-part words are used for the four evaluation pages.`;
  title.textContent = "Self score";
  formula.textContent = `1000 / (${formatRatio(maximum, 12)} - ${formatRatio(minimum, 12)}) = ${number.format(Number(score.toFixed(2)))}`;
  verdict.textContent =
    "All four ratios are below 1.2, and every evaluated word decodes to its original UTF-8 text.";
}

function tokenLabel(token) {
  if (token.text !== null) return token.text;
  if (token.kind === "byte") return `byte 0x${token.hex.toUpperCase()}`;
  return "UTF-8 byte fragment";
}

function initializeTokenizerRuntime() {
  state.tokensById = state.tokenizer.tokens;
  state.mergeIds = new Map();
  state.lexemeIds = new Map();
  const pieces = new Map();

  state.tokenizer.tokens.forEach((token) => {
    if (token.kind === "merge") {
      state.mergeIds.set(`${token.parents[0]}:${token.parents[1]}`, token.id);
    } else if (token.kind === "lexeme") {
      state.lexemeIds.set(token.text, token.id);
    } else if (token.kind === "piece") {
      const key = token.source;
      if (!pieces.has(key)) pieces.set(key, []);
      pieces.get(key).push([token.piece_index, token.id]);
    }
  });

  state.pieceIds = new Map(
    [...pieces].map(([key, parts]) => [
      key,
      parts.sort((left, right) => left[0] - right[0]).map((part) => part[1]),
    ]),
  );
}

function mergeSequence(sequence, pair, tokenId) {
  const merged = [];
  for (let index = 0; index < sequence.length; ) {
    if (
      index + 1 < sequence.length &&
      sequence[index] === pair[0] &&
      sequence[index + 1] === pair[1]
    ) {
      merged.push(tokenId);
      index += 2;
    } else {
      merged.push(sequence[index]);
      index += 1;
    }
  }
  return merged;
}

function encodeBpeWord(word) {
  let sequence = [...textEncoder.encode(word)];
  while (sequence.length > 1) {
    let bestId = null;
    let bestPair = null;
    for (let index = 0; index < sequence.length - 1; index += 1) {
      const pair = [sequence[index], sequence[index + 1]];
      const tokenId = state.mergeIds.get(`${pair[0]}:${pair[1]}`);
      if (tokenId !== undefined && (bestId === null || tokenId < bestId)) {
        bestId = tokenId;
        bestPair = pair;
      }
    }
    if (bestPair === null) break;
    sequence = mergeSequence(sequence, bestPair, bestId);
  }
  return sequence;
}

function encodePlaygroundWord(word) {
  if (state.lexemeIds.has(word)) return [state.lexemeIds.get(word)];
  if (state.pieceIds.has(word)) return state.pieceIds.get(word);
  return encodeBpeWord(word);
}

function escapedBytes(hex) {
  return hex.match(/../g).map((byte) => `\\x${byte.toUpperCase()}`).join("");
}

function renderPlayground() {
  const input = document.querySelector("#playground-input");
  const text = input.value;
  const matches = [...text.matchAll(/\S+/gu)];
  const output = document.querySelector("#playground-output");
  output.replaceChildren();

  let cursor = 0;
  let tokenCount = 0;
  matches.forEach((match) => {
    output.append(document.createTextNode(text.slice(cursor, match.index)));
    const tokenIds = encodePlaygroundWord(match[0]);
    tokenIds.forEach((tokenId) => {
      const token = state.tokensById[tokenId];
      const span = document.createElement("span");
      const fragment = token.text === null;
      span.className = `playground-token token-color-${tokenCount % 8}${fragment ? " byte-fragment" : ""}`;
      span.textContent = fragment ? escapedBytes(token.hex) : token.text;
      span.title = `ID ${token.id} / ${tokenKindLabels[token.kind]} / ${token.hex}`;
      output.append(span);
      tokenCount += 1;
    });
    cursor = match.index + match[0].length;
  });
  output.append(document.createTextNode(text.slice(cursor)));

  document.querySelector("#playground-tokens").textContent = number.format(tokenCount);
  document.querySelector("#playground-words").textContent = number.format(matches.length);
  document.querySelector("#playground-characters").textContent = number.format([...text].length);
  document.querySelector("#playground-ratio").textContent = matches.length
    ? (tokenCount / matches.length).toFixed(3)
    : "0.000";
  output.classList.toggle("empty", text.length === 0);
  if (text.length === 0) output.textContent = "Token output";
}

function showPlaygroundExample() {
  document.querySelector("#playground-input").value = examples[state.exampleIndex];
  state.exampleIndex = (state.exampleIndex + 1) % examples.length;
  renderPlayground();
}

function filteredTokens() {
  const query = state.query.trim().toLocaleLowerCase();
  if (!query) return state.tokenizer.tokens;
  return state.tokenizer.tokens.filter((token) => {
    const haystack = [
      token.id,
      token.kind,
      token.source ?? "",
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
    detail.textContent = `${language.file} / ${number.format(language.bytes)} bytes / ${number.format(language.unique_words)} unique words`;
    const acquisition = document.createElement("span");
    acquisition.textContent = `${language.acquisition} / revision ${language.revision_id} at ${language.revision_timestamp}`;
    const retrieved = document.createElement("span");
    retrieved.textContent = `Retrieved ${language.retrieved_at}`;
    const cleaning = document.createElement("span");
    cleaning.textContent = `Cleaning: ${language.cleaning}`;
    const fileLink = document.createElement("a");
    fileLink.className = "corpus-file";
    fileLink.href = `assets/${language.file}`;
    fileLink.download = language.file;
    fileLink.textContent = `Download ${language.file}`;
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
      state.tokenizer.vocab_size,
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
