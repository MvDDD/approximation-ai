class NeuralNet {
	constructor(layers, act=(a)=>Math.max(a,0), out=(a)=>Math.tanh(a)) {
		if (layers) {
			if (layers.nodes) {
				this.nodes = layers.nodes
				this.paths = layers.paths
				this.act = layers.act
				this.out = layers.out
			} else {
				this.nodes = layers.map(i => Array(i).fill(0).map(a => Math.random() - 0.5))
				this.act = act
				this.out = out
				this.paths = []
				for (let layer = 0; layer < layers.length - 1; layer++) {
					let pathLayer = []
					for (let start = 0; start < layers[layer]; start++) {
						for (let end = 0; end < layers[layer + 1]; end++) {
							pathLayer.push([start, end, Math.random() - 0.5])
						}
					}
					this.paths.push(pathLayer)
				}
			}

		}
	}
	mutatePaths(amount) {
		this.paths = this.paths.map(layer => layer
			.map(path => [path[0], path[1], path[2] + ((Math.random() - 0.5) * amount)])
		)
	}
	mutateNodes(amount) {
		this.nodes = this.nodes.map(layer => layer.map(node => node + ((Math.random() - 0.5) * amount)))
	}
	clone() {
		return new NeuralNet({ nodes: this.nodes.map(l => l.slice()), paths: this.paths.map(l => l.slice()), act:this.act, out:this.out})
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
		return { nodes: this.nodes.map(l => l.slice()), paths: this.paths.map(l => l.slice()), act:this.act.toString(), out:this.out.toString()}
	}
	fromString(str) {
		let m = JSON.parse(str)
		m.act = eval("(()=>return " + m.act + ")()")
		m.out = eval("(()=>return " + m.out + ")()")
		return new NeuralNet(m)
	}
	draw(ctx) {
		const layerWidth = (ctx.canvas.width - 80) / (this.nodes.length + 1);
		const radius = 20; // Radius of the nodes

		let nodePositions = [];

		// Draw the nodes
		for (let layer = 0; layer < this.nodes.length; layer++) {
			const layerHeight = (ctx.canvas.height - 80) / (this.nodes[layer].length + 1);
			let currentLayerPositions = [];

			for (let node = 0; node < this.nodes[layer].length; node++) {
				const x = ((layer + 1) * layerWidth) + 40;
				const y = ((node + 1) * layerHeight) + 40;
				currentLayerPositions.push({ x, y });

				// Draw the node
				ctx.beginPath();
				ctx.arc(x, y, Math.abs(this.nodes[layer][node]) * 10, 0, Math.PI * 2);
				ctx.fillStyle = "#3498db";
				ctx.fill();
				ctx.stroke();
			}

			nodePositions.push(currentLayerPositions);
		}

		// Draw the paths (connections)
		ctx.strokeStyle = "#2c3e50";
		for (let layer = 0; layer < this.paths.length; layer++) {
			for (let path of this.paths[layer]) {
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