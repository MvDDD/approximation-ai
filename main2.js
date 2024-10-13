const populationSize = 50;
let generations = Infinity; // Number of generations to evolve
const octaves = 20
const canvas = document.getElementById('canvas');
const mcanvas = document.getElementById('mcanvas')
const ctx2 = mcanvas.getContext("2d")
const ctx = canvas.getContext('2d');

const netconfig = [
    fourierSeries([0], octaves).length,
    5,
/*    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,*/
    5,
    5,
    5,
    1
    ]


function genFunc(x){
    return Math.max(Math.min(Math.tan(x/4), 0.9), -0.9)
}

function mrate(x){
    return Math.max(x / 120, 0.0001)
}
function calcloss(output, target) {
    return Math.pow(output - target, 2);
}

function generateData() {
    const data = [];
    for (let i = 0; i < 100; i+=0.5) {
        let x = Math.random() * Math.PI * 2 - Math.PI; // Random x in range [-π, π]
        let y = genFunc(x); // Target output for the formula (y = sin(x))
        data.push({ x, y });
    }
    return data;
}

function fourierSeries(inputs, octaves) {
    let out = [];
    inputs.forEach(i => {
        for (let j = 1; j <= octaves; j++) {
            //out.push(i * (2 ** j))

            out.push(i * Math.sin(j)); // Sine component
            out.push(i * Math.cos(j)); // Cosine component
            
        }
    });
    return [1, ...out]; // Include bias term
}

// Define the loss function (Mean Squared Error)

let net = new NeuralNet(
    netconfig, // Input size, hidden layer size, output size
    x => Math.max(0.1*x, x), // ReLU activation function
    a => Math.tanh(a) // Output activation function (squashing output between -1 and 1)
    );
net.max = 20

let loss = 0.01
let gen = 0;
let population = Array.from({ length: populationSize }, () => {
    let n = net.clone();
    n.mutateNodes(loss);
    n.mutatePaths(loss);
    return { n, p: 0 }; // p will store the error for each network
});

const trainingData = generateData(); // Generate data for the formula (sin(x))

async function evolveFormula() {
    // Reset the population error for the current generation
    population.forEach(p => p.p = 0);

    // Calculate the fitness (error) of each network in the population
    trainingData.forEach(({ x, y }) => {
        let inputs = fourierSeries([x], octaves); // Fourier series input for x
        population.forEach(p => {
            let output = p.n.run(inputs)[0]; // Network's prediction
            p.p += calcloss(output, y); // Accumulate the error (loss)
        });
    });

    // Sort population by fitness (lower error is better)
    population.sort((a, b) => a.p - b.p);

    // Keep the best performing network
    let bestNet = population[0];

    // Mutate the population for the next generation
    population = population.map(() => {
        let n = bestNet.n.clone();
        n.mutateNodes(mrate(bestNet.p)); // Mutate nodes slightly (with fallback mutation strength)
        n.mutatePaths(mrate(bestNet.p)); // Mutate paths slightly
        return { n, p: 0 };
    });
    //population.pop()
    //population.push({n:bestNet.n, p:0})

    // Draw the network's approximation
    drawCurve(bestNet.n, gen); // Update the drawing every generation
    console.log(bestNet.n.nodes);

    // Continue to the next generation
    if (gen < generations) {
        gen++;
        requestAnimationFrame(evolveFormula);
    }
}

// Function to draw the actual sine curve and the network's approximation
function drawCurve(network, generation) {
    let normalised = new NeuralNet().fromString(network.toString());
    let maxNode = Math.max(...normalised.nodes.flat());
    let maxPath = Math.max(...normalised.paths.flat(2).map(p => p[2]));

    // Prevent division by zero
    maxNode = maxNode;
    maxPath = maxPath;

    normalised.nodes = normalised.nodes.map(l => l.map(n => (n / maxNode) * 20));
    normalised.paths.forEach(layer => layer.forEach(path => path[2] = (path[2] / maxPath) * 20));

    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    ctx2.clearRect(0, 0, mcanvas.width, mcanvas.height); // Clear the canvas

    // If the network has a 'draw' method to visualize its structure, call it
    normalised.draw(ctx2);

    // Draw the true sine function (in blue)
    ctx.beginPath();
    ctx.strokeStyle = 'blue';
    for (let i = 0; i < canvas.width; i++) {
        let x = (i / canvas.width) * 2 * Math.PI - Math.PI; // Scale x to [-π, π]
        let y = genFunc(x); // True value of sin(x)
        let screenX = i;
        let screenY = canvas.height / 2 - y * (canvas.height / 2); // Scale y to fit canvas
        if (i === 0) {
            ctx.moveTo(screenX, screenY);
        } else {
            ctx.lineTo(screenX, screenY);
        }
    }
    ctx.stroke();

    // Draw the neural network's approximation (in red)
    ctx.beginPath();
    ctx.strokeStyle = 'red';
    for (let i = canvas.width; i >0; i--) {
        let x = (i / canvas.width) * 2 * Math.PI - Math.PI; // Scale x to [-π, π]
        let inputs = fourierSeries([x], octaves); // Use Fourier series for the input
        let y = network.run(inputs)[0]; // Network's prediction
        let screenX = i;
        let screenY = canvas.height / 2 - y * (canvas.height / 2); // Scale y to fit canvas
        if (i === 0) {
            ctx.moveTo(screenX, screenY);
        } else {
            ctx.lineTo(screenX, screenY);
        }
    }
    ctx.stroke();

    // Display the current generation
    ctx.fillStyle = 'black';
    ctx.font = '20px Arial';
    ctx.fillText(`Generation: ${generation}`, 10, 30);
    document.getElementById("outmodel").value = network.toString()
}

// Start evolving
evolveFormula();
