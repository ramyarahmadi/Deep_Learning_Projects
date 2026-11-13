# Gradient Descent: A Ball on a Rugged Terrain

A ball starts at a random point on a 3D terrain and rolls downhill using gradient descent until it reaches the bottom of a valley.

## Why I did this project

In machine learning, a model has parameters, and every choice of parameters gives some error. If we draw the error as a height, we get a surface. Training a model means walking down this surface to find low error. I wanted to **see** this process instead of only reading the formula.

## The idea in simple words

- The terrain is the error surface. A point on the ground is a pair of parameters (a, b). The height at that point is the error.
- The gradient is the slope at the ball's position. It points uphill.
- The ball moves a small step in the opposite direction, so it goes downhill.
- The learning rate is the size of each step.

## How it works

1. **Load the terrain** (the Jacksboro fault elevation data from matplotlib) and use the elevation as the error.
2. **Rescale** a, b and the error to the range 0 to 1. Why: degrees and meters have very different sizes, and one learning rate would not fit both directions.
3. **Build the error function `E(a, b)`** with interpolation. Why: the terrain is a grid of numbers, and the ball can stand between grid points.
4. **Estimate the gradient** with finite differences. Why: we have no formula for this terrain, so we measure how the height changes when we move a tiny bit.
5. **Pick a random start.** Why: this shows that different starting points can end in different valleys.
6. **Run the training loop**: forward (read the height), loss (save it), backward (compute the slope), update (step downhill).
7. **Plot** the loss curve, the path in 3D and the path on the contour lines.

The loop follows the same structure as my linear regression exercise (initializer, forward, loss, backward, update). There are no mini-batches here, because the error comes from the terrain and not from data samples.

## Results

- The error goes down quickly at the beginning and then slows down near the bottom.
- Different random starts end in different valleys. The ball finds a **local minimum**, and not always the lowest point of the whole terrain.
- A learning rate that is too large makes the ball jump around. A learning rate that is too small makes it very slow.

## What I learned

- Gradient descent only uses local information: the slope where the ball stands now.
- The starting point matters when the surface has many valleys.
- Rescaling the inputs makes training easier.

## How to run

```bash
pip install numpy matplotlib scipy jupyter
jupyter notebook ball_on_terrain.ipynb
```

Run all cells. Run the random initializer cell and the cells after it again to try a new start.

## Files

- `ball_on_terrain.ipynb`: the full project
- `gradient_descent_ball.py`: a first, shorter script version

## Ideas for next steps

- Add momentum so the ball can pass over small bumps.
- Try several random starts in one plot.
- Compare different learning rates.
