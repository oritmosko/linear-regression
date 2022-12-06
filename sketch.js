let x_vals = [];
let y_vals = [];

let a_tensor, b_tensor;

const optimizer = tf.train.adam(0.5);

function setup() {
  createCanvas(400, 400);
  // ax+b - random initialization
  a_tensor = tf.variable(tf.scalar(random(1)));
  b_tensor = tf.variable(tf.scalar(random(1)));
}

function loss(predictions_tensor, labels_tensor) {
  return predictions_tensor.sub(labels_tensor).square().mean();
}

function predict(xs_tensor) {
  return xs_tensor.mul(a_tensor).add(b_tensor);
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function draw() {
  if (x_vals.length > 0) {
    const xs_tensor = tf.tensor1d(x_vals);
    const ys_tensor = tf.tensor1d(y_vals);
    // updates a_tensor, b_tensor according to all the clicks by now
    optimizer.minimize(() => loss(predict(xs_tensor), ys_tensor));
  }

  background(0);
  
  // re-draw the points
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    point(px, py);
  }


  // draw the predicted line
  const x01_tensor = tf.tensor1d([0, 1]);

  const y01 = predict(x01_tensor).dataSync();

  let x1 = map(0, 0, 1, 0, width);
  let x2 = map(1, 0, 1, 0, width);

  let y1 = map(y01[0], 0, 1, height, 0);
  let y2 = map(y01[1], 0, 1, height, 0);

  strokeWeight(2);
  line(x1, y1, x2, y2);
}

function renderNew() {
	push();
	background(0);
  x_vals = [];
  y_vals = [];
  a_tensor = tf.variable(tf.scalar(random(1)));
  b_tensor = tf.variable(tf.scalar(random(1)));
	pop();
}
