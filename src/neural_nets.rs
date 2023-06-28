use nalgebra::{DMatrix, DVector};

pub fn sigmoid(value: f32) -> f32{
	return 1.0 / (1.0 + libm::expf(-value));
}

pub fn disigmoid(value: f32) -> f32{
	return value * (1.0 - value);
}

pub struct Layer{
	layer_weights: DMatrix<f32>,
	layer_bias: DVector<f32>,
	layer_inputs: DVector<f32>,
	layer_outputs: DVector<f32>
}

pub fn create_layer(num_inputs: usize, num_nodes: usize) -> Layer {
	let mut rng = rand::thread_rng();
    let dist = rand::distributions::Standard;

	return Layer { 
		layer_weights: DMatrix::from_distribution(num_nodes, num_inputs, &dist, &mut rng), 
		layer_bias: DVector::from_distribution(num_nodes, &dist, &mut rng),
		layer_inputs: DVector::identity(num_inputs),
		layer_outputs: DVector::identity(num_nodes) 
	}
}	


pub struct NeuralNet{
    num_inputs: usize,
	num_outputs: usize,
	layers: Vec<(usize, Layer)>,
	accumelated_cost: f32,
	num_calcs: i32
}

impl NeuralNet {
	pub fn debug_print(&self){
		println!("{} accumelated_cost, {} num_inputs, {} num_outputs, {} num_calcs",
		 self.accumelated_cost,
		 self.num_inputs, 
		 self.num_outputs,
		 self.num_calcs);
	}

	pub fn feed_forward(&mut self, initial_input: &DVector<f32>) -> DVector<f32> {
		let mut input = initial_input.clone();
		let mut output = DVector::identity(1);
		
		for layer in self.layers.iter_mut(){
			layer.1.layer_inputs = input.clone();
			
			output = &layer.1.layer_weights * input;
			output = output + &layer.1.layer_bias;
			output = output.map(sigmoid);

			layer.1.layer_outputs = output.clone();
			input = output.clone();
		}

		return output;
	}
}

pub fn create_neural_net(num_inputs: usize, num_outputs: usize, num_layers: usize, num_layer_nodes: usize) -> NeuralNet {
	let mut net = NeuralNet{
		num_inputs: num_inputs,
		num_outputs: num_outputs,
		layers: Vec::new(),
		accumelated_cost: 0.0,
		num_calcs: 0

	};

	for n in 0..num_layers {
		if n == 0 {
			net.layers.push((num_layer_nodes, create_layer(num_inputs, num_layer_nodes)));
		}
		else {
			net.layers.push((num_layer_nodes, create_layer(num_layer_nodes, num_layer_nodes)));			
		}
	} 

	net.layers.push((num_outputs, create_layer(num_layer_nodes, num_outputs)));	

	return net;
}