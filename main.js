let populationSize = 50;
let o = 2

function fourierSeries(inputs, octaves) {
    let out = [];
    inputs.forEach(i => {
        for (let j = 1; j < octaves; j++) {

        let s = i * Math.sin(j)
        let c = i * Math.cos(j)
            out.push(s,c,s*c,s/(c+0.00001));
        }
    });
    out.push(Math.sin(inputs[0])); // Emphasize x-direction (horizontal)
    out.push(Math.cos(inputs[0])); 
    out.push(Math.sin(inputs[1])); // Emphasize y-direction (vertical)
    out.push(Math.cos(inputs[1]));
    return [...out];
}

let net = new NeuralNet(
    [
        fourierSeries([0, 0], o).length,
        5,
        10,
        1,
        
    ],
    (a) => (Math.max(0.1 * a, a)), // Changed from 0.1 * a to Math.max(0.1 * a, a) for clarity
    (a) => (Math.tanh(a))
);

let imgI = new Image();
imgI.src = "./img1.png";
let dataI = [];
let ctx;
let loss = 1

imgI.onload = () => {
	try{
    let canvas = document.getElementById("canvas");
    canvas.width = imgI.width;
    canvas.height = imgI.height;
    ctx = canvas.getContext("2d");
    ctx.drawImage(imgI, 0, 0);
    let imgData = ctx.getImageData(0, 0, imgI.width, imgI.height); // Changed from img4 to imgI
    for (let i = 0; i < imgData.data.length; i += 4) { // Fixed to iterate over imgData.data
        dataI.push(imgData.data[i] > 128 ? 1 : -1); // Accessing imgData.data for pixel values
    }
    train(); // Uncommented train function call
	}catch(e){console.log(e)}
}
let midloss = 0
async function train() {
    let population = Array.from({ length: populationSize }, () => { // Changed to Array.from for better readability
        let n = net.clone(); // Fixed to invoke clone method
        n.mutateNodes(midloss);
        n.mutatePaths(midloss);
        return { n, p: 0 };
    });

    let mloss = 0;
    for (let x = 0; x < imgI.width; x++) {
        console.log(x);
        for (let y = 0; y < imgI.height; y++) {
            const mappedX = (x / imgI.width) * (2 * Math.PI) -  Math.PI;
            const mappedY = (y / imgI.height) * (2 * Math.PI) +  Math.PI;
            let inputs = fourierSeries([mappedX, mappedY], o);
            let expected = dataI[y * imgI.width + x];
            population.forEach(p => {
                let out = p.n.run(inputs)[0];
                p.p += Math.abs(out - expected); // Accumulate error for each individual
            });
            mloss += Math.abs(net.run(inputs)[0] - expected); // Total loss for current pixel
        }
    }

    population = population.sort((a, b) => a.p - b.p)[0]; // Get the best individual
    loss = (mloss/(imgI.width*imgI.height))**2
    if (population.p < mloss) {
    	midloss = (population.p/(imgI.width*imgI.height))**2
        net = population.n.clone(); // Clone best individual into the main network
    }
    draw();
    requestAnimationFrame(train); // Request next animation frame for training
}

function mapColor(t) {
    t = Math.max(-1, Math.min(1, t)); // Clamping the value of t between -1 and 1
    let r, g, b;

    if (t < 0) {
        const factor = (t + 1) / 1; 
        r = Math.floor(0 + factor * (255 - 0)); 
        g = Math.floor(0 + factor * (255 - 0)); 
        b = 0; 
    } else if (t < 0.5) {
        const factor = (t - 0) / 0.5; 
        r = Math.floor(255 + factor * (128 - 255)); 
        g = Math.floor(255 + factor * (0 - 255)); 
        b = Math.floor(0 + factor * (128 - 0)); 
    } else {
        const factor = (t - 0.5) / 0.5; 
        r = Math.floor(128 + factor * (255 - 128)); 
        g = Math.floor(0 + factor * (255 - 0)); 
        b = Math.floor(128 + factor * (255 - 128)); 
    }

    return `rgb(${r}, ${g}, ${b})`;
}

async function draw() {
    for (let x = 0; x < imgI.width; x++) {
        for (let y = 0; y < imgI.height; y++) {
            const mappedX = (x / imgI.width) * (4 * Math.PI) - 2 * Math.PI;
            const mappedY = (y / imgI.height) * (4 * Math.PI) + 2 * Math.PI;
            let inputs = fourierSeries([mappedX, mappedY], o);
            ctx.fillStyle = mapColor(net.run(inputs)[0]);
            ctx.fillRect(x, y, 1, 1);
        }
    }
}