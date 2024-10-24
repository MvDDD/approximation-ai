class NeuralNet {
	constructor(layers, act=(a)=>Math.max(a,0), out=(a)=>Math.tanh(a), random = Math.random) {
		if (layers) {
			if (layers.nodes) {
				this.nodes = layers.nodes
				this.paths = layers.paths
				this.act = layers.act
				this.out = layers.out
				this.random = random
			} else {
				this.nodes = layers.map(i => 
					Array(i)
					.fill(0)
					.map((a) => 
						0
						))
				this.act = act
				this.out = out
				this.random = random
				this.paths = []
				for (let layer = 0; layer < layers.length - 1; layer++) {
					let pathLayer = []
					for (let start = 0; start < layers[layer]; start++) {
						for (let end = 0; end < layers[layer + 1]; end++) {
							pathLayer.push([start, end, 1/(layers[layer]*layers[layer + 1])/*layers[layer]*/])
						}
					}
					this.paths.push(pathLayer)
				}
			}

		}
	}
	mutatePaths(amount) {
		if (!this.limit){
			this.paths = this.paths.map(layer => layer
				.map(path => [path[0], path[1], path[2] + ((this.random() - 0.5) * amount)])
				)
		} else {
			this.paths = this.paths.map(layer => layer
				.map(path => {
					let newWeight = path[2] + ((this.random() - 0.5) * amount);
				// Ensure the weights don't grow too large
					newWeight = Math.max(Math.min(newWeight, this.max), -this.max);
					return [path[0], path[1], newWeight];
				})
				);
		}
		console.log(this.paths)
	}

	// Mutation for nodes with limits to prevent large mutations
	mutateNodes(amount) {
		if (!this.max){
			this.nodes = this.nodes.map(layer => layer.map(node => node + ((this.random() - 0.5) * amount)))
		} else {
			this.nodes = this.nodes.map(layer => layer.map(node => {
				let newNode = node + ((this.random() - 0.5) * amount);
			// Ensure nodes don't grow too large
				newNode = Math.max(Math.min(newNode, this.max), -this.max);
				return newNode;
			}));
		}
		console.log(this.nodes)
	}
	clone() {
		let n = new NeuralNet({ nodes: this.nodes.map(l => l.slice()), paths: this.paths.map(l => l.map(p=>p.slice())), act:this.act, out:this.out})
		if (this.max){
			n.max = this.max
		}
		return n
	}
	run(inputs) {
		if (inputs.length !== this.nodes[0].length) {
			throw new TypeError("input size incorrect: " + inputs.size)
		}
		let model = this.clone()
		inputs.forEach((i, j) => model.nodes[0][j] = i)
		model.paths.forEach((layer, layerNum) => {
			layer.forEach(path => {
				model.nodes[layerNum + 1][path[0]] += model.nodes[layerNum][path[1]] * path[2]
			})
			model.nodes[layerNum+1] = model.nodes[layerNum+1].map(a=>this.act(a))
		})
		return model.nodes.pop().map(a=>this.out(a))
	}

	toString() {
		return JSON.stringify({ nodes: this.nodes.map(l => l.map(v=>parseFloat(v.toFixed(10)))), paths: this.paths.map(l => l.map(p=>{return [p[0], p[1], parseFloat(p[2].toFixed(10))]})), act:this.act.toString(), out:this.out.toString()})
	}
	fromString(str) {
		let m = JSON.parse(str)
		m.act = eval("(()=>{return " + m.act + "})()")
		m.out = eval("(()=>{return " + m.out + "})()")
		return new NeuralNet(m)
	}
	from(net){
		return net.clone()
	}
	draw(ctx, size = 20) {
		let normalised = this.clone()
		let maxNode = Math.max(...normalised.nodes.flat());
		let maxPath = Math.max(...normalised.paths.map(l=>l.map(p => p[2])).flat());
		normalised.nodes = normalised.nodes.map(l => l.map(n => (n / maxNode) * size));
		normalised.paths.forEach(layer => layer.forEach(path => path[2] = (path[2] / maxPath) * size/4));

		const layerWidth = (ctx.canvas.width - 80) / (normalised.nodes.length + 1);
		const radius = 20; // Radius of the nodes

		let nodePositions = [];

		// Draw the nodes
		for (let layer = 0; layer < normalised.nodes.length; layer++) {
			const layerHeight = (ctx.canvas.height - 80) / (normalised.nodes[layer].length + 1);
			let currentLayerPositions = [];

			for (let node = 0; node < normalised.nodes[layer].length; node++) {
				const x = ((layer + 1) * layerWidth) + 40;
				const y = ((node + 1) * layerHeight) + 40;
				currentLayerPositions.push({ x, y });

				// Draw the node
				ctx.lineWidth = 2
				ctx.beginPath();
				ctx.arc(x, y, Math.abs(normalised.nodes[layer][node]), 0, Math.PI * 2);
				ctx.fillStyle = "#3498db";
				ctx.fill();
				ctx.stroke();
			}

			nodePositions.push(currentLayerPositions);
		}

		// Draw the paths (connections)
		ctx.strokeStyle = "#2c3e50";
		for (let layer = 0; layer < normalised.paths.length; layer++) {
			for (let path of normalised.paths[layer]) {
				const start = nodePositions[layer][path[0]];
				const end = nodePositions[layer + 1][path[1]];
				ctx.lineWidth = Math.abs(path[2])

				ctx.beginPath();
				ctx.moveTo(start.x, start.y);
				ctx.lineTo(end.x, end.y);
				ctx.stroke();
			}
		}
	}
}


if (typeof self == "undefined"){
	module.exports = {NeuralNet}
}
