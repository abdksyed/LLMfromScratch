export function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export function relu(value) {
  return Math.max(0, value);
}

export function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

export function formatFixed(value, digits = 3) {
  return value.toFixed(digits).replace("-0.000", "0.000");
}

export function multiplyMatrices(left, right) {
  return left.map((leftRow) =>
    right[0].map((_, colIndex) => {
      let sum = 0;
      for (let i = 0; i < right.length; i++) sum += leftRow[i] * right[i][colIndex];
      return sum;
    })
  );
}

export function multiplyMatrixVector(matrix, vector) {
  return matrix.map((row) => dot(row, vector));
}

export function addVectors(left, right) {
  return left.map((value, index) => value + right[index]);
}
