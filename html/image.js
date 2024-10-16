let populationSize = 10; // Size of the population for the genetic algorithm
let o = 1; // Number of octaves for the Fourier series

function calcLoss(x) {
    return x
}

// Fourier series function using sine and cosine transformations
function fourierSeries(inputs, octaves) {
    let out = [];
    inputs.forEach(i => {
        for (let j = 1; j <= octaves; j++) { // Fixed to include all octaves, inclusive
            inputs.forEach(k => {
            out.push(
            Math.sin((j *  i)+1),
            Math.cos((j *  i)+1),
            //Math.tanh(j * i)
            ); // Add sine and cosine values to output
            })
        }
    });
    return [...out, 1];
}

// Neural network configuration
let net = new NeuralNet(
    [
        fourierSeries([0, 0], o).length, // Input size from Fourier series
        5,
        3, // Output layer (RGB values have 3 channels)
    ],
    (a) => Math.max(0.1 * a, a), // Leaky ReLU activation function
    (a) => Math.tanh(a) // Hyperbolic tangent activation function
);

let imgI = new Image();
imgI.src = "download.png"

let dataI = []; // Array to store image pixel data
let ctx;
let loss = 1;

imgI.onload = () => {
    try {
        let canvas = document.getElementById("canvas");
        canvas.width = imgI.width;
        canvas.height = imgI.height;
        ctx = canvas.getContext("2d");
        ctx.drawImage(imgI, 0, 0);

        let imgData = ctx.getImageData(0, 0, imgI.width, imgI.height); 
        for (let i = 0; i < imgData.data.length; i += 4) {
            // Normalize the RGB values to range [-π, π]
            dataI.push([
                (imgData.data[i] / 255) * Math.PI * 2 - Math.PI,
                (imgData.data[i + 1] / 255) * Math.PI * 2 - Math.PI,
                (imgData.data[i + 2] / 255) * Math.PI * 2 - Math.PI,
            ]);
        }
        train(); // Start training after image is loaded
    } catch (e) {
        console.error(e); // Corrected console logging
    }
}

let midloss = 1;

// Training function
async function train() {
    console.log("frame");

    let population = Array.from({ length: populationSize }, () => {
        let n = net.clone(); // Clone the neural network for mutation
        n.mutateNodes(midloss || 0.001); // Mutate nodes
        n.mutatePaths(midloss || 0.001); // Mutate connections
        return { n, p: 0 }; // p is the fitness score (lower is better)
    });

    let mloss = 0;

    // Loop over image pixels
    for (let x = 0; x < imgI.width; x++) {
        for (let y = 0; y < imgI.height; y++) {
            const mappedX = (x / imgI.width) * (2 * Math.PI) - Math.PI;
            const mappedY = (y / imgI.height) * (2 * Math.PI) - Math.PI;
            let inputs = fourierSeries([mappedX, mappedY], o); // Fourier-transformed inputs
            let expected = dataI[y * imgI.width + x]; // Expected pixel values

            population.forEach(p => {
                let out = p.n.run(inputs); // Network output for current pixel
                p.p += Math.abs(out[0] - expected[0]) + Math.abs(out[1] - expected[1]) + Math.abs(out[2] - expected[2]) // Calculate fitness (error)
            });
        }
    }

    // Select the best-performing network
    let bestIndividual = population.sort((a, b) => a.p - b.p)[0];
    midloss = calcLoss(bestIndividual.p / (imgI.width * imgI.height)); // Fix midloss calculation
    net = bestIndividual.n.clone(); // Set the best network as the main network

    draw(); // Draw the new frame
    requestAnimationFrame(train); // Continue training after a delay
}

// Map normalized values [-1, 1] to RGB color values [0, 255]
// Map normalized values [-1, 1] to RGB color values [0, 255]
function mapColor(r, g, b) {
    r = Math.max(-1, Math.min(1, r)); // Ensure value is in the range [-1, 1]
    g = Math.max(-1, Math.min(1, g));
    b = Math.max(-1, Math.min(1, b));

    r = (r + 1) / 2; // Scale to [0, 1]
    g = (g + 1) / 2;
    b = (b + 1) / 2;

    return `rgb(${r * 255}, ${g * 255}, ${b * 255})`;
}

// Neural network's drawing function
async function draw() {
    document.getElementById("net").value = net.toString();
    net.draw(document.getElementById("mcanvas").getContext("2d"))
    for (let x = 0; x < imgI.width; x++) {
        for (let y = 0; y < imgI.height; y++) {
            const mappedX = (x / imgI.width) * (2 * Math.PI) - Math.PI;
            const mappedY = (y / imgI.height) * (2 * Math.PI) - Math.PI;
            let inputs = fourierSeries([mappedX, mappedY], o);
            let outputColor = net.run(inputs);
            ctx.fillStyle = mapColor(...outputColor); 
            ctx.fillRect(x, y, 1, 1); 
        }
    }
}
