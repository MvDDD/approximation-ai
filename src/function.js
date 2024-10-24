const populationSize = 40;
let generations = Infinity; // Number of generations to evolve
const octaves = 10

let seed = Date.now()
function random(){
    seed ^= seed << 5
    seed ^= seed >> 4
    seed ^= seed << 7
    return (seed%100000000)/100000000
}
const canvas = document.getElementById('canvas');
const mcanvas = document.getElementById('mcanvas')
const ocanvas = document.getElementById('out')
const ctx3 = ocanvas.getContext("2d")
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
    5,*/
    5,
    5,
    5,
    5,
    1
    ]


function genFunc(x){
    return Math.cos(x)/2
}

function mrate(x){
    return ((1+(x/200))**2)-1 || 0.0000001
}

function calcloss(output, target) {
    return Math.pow(Math.abs(output - target), 1.5);
}

function generateData() {
    const data = [];
    for (let i = 0; i < 100; i+=15) {
        let x = (i/100) * Math.PI * 2 - Math.PI; // Random x in range [-π, π]
        let y = genFunc(x); // Target output for the formula (y = sin(x))
        data.push({ x, y:genFunc(x) });
    }
    return data;
}

function fourierSeries(inputs, octaves) {
    let out = [];
    inputs.forEach(i => {
        for (let j = 0; j <= octaves; j++) {
            //out.push(i * (2 ** j))
            //out.push(i * Math.sqrt(j))
            out.push(j * i)
            out.push(j * Math.sin(i*j)); // Sine component
            out.push(j * Math.cos(i*j)); // Cosine component
            
        }
    });
    return [1, ...out]; // Include bias term
}

// Define the loss function (Mean Squared Error)

let net = new NeuralNet(
    netconfig, // Input size, hidden layer size, output size
    x => Math.max(0.6*x, x), // ReLU activation function
    a => Math.tanh(a), // Output activation function (squashing output between -1 and 1)
    random
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
    loss = bestNet.p

    // Mutate the population for the next generation
    population = population.map(() => {
        let n = bestNet.n.clone();
        n.mutateNodes(mrate(loss)); // Mutate nodes slightly (with fallback mutation strength)
        n.mutatePaths(mrate(loss)); // Mutate paths slightly
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
function circle(ctx, sin, centerX, centerY, radius, segments = 100) {
    ctx.beginPath();  // Start drawing the path

    for (let i = 0; i <= segments; i++) {
        // Calculate angle for current segment
        let angle = ((i / segments) * 2 * Math.PI)%(2*Math.PI)
        // Use sin for both x and y, with an offset for cosine
        let x = centerX + radius * sin((angle + Math.PI / 2)%(2*Math.PI));  // cos(θ) = sin(θ + π/2)
        let y = centerY + radius * sin(angle%(2*Math.PI));

        // Move to the first point or draw a line to the next point
        if (i === 0) {
            ctx.moveTo(x, y);  // Move to the start of the circle
        } else {
            ctx.lineTo(x, y);  // Draw a line to the calculated point
        }
    }

    //ctx.closePath();  // Close the path to form a circle
    ctx.stroke();     // Stroke the circle (draw outline)
}


function drawCurve(network, generation) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas
    ctx2.clearRect(0, 0, mcanvas.width, mcanvas.height); // Clear the canvas
    ctx3.clearRect(0, 0, ocanvas.width, ocanvas.height); // Clear the canvas
    ctx3.strokeStyle = "blue"
    circle(ctx3,(x)=>genFunc(x/2)*2, 500, 500, 480, 100)
    ctx3.strokeStyle = "red"
    circle(ctx3,(a)=>network.run(fourierSeries([a/2], octaves))[0]*2, 500, 500, 480, 100)
    // If the network has a 'draw' method to visualize its structure, call it
    network.draw(ctx2);

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
    document.getElementById("loss").value = loss
    document.getElementById("outmodel").value = network.toString()
}

// Start evolving
evolveFormula();
