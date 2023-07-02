use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

pub fn sigmoid(value: f32) -> f32{
	return 1.0 / (1.0 + libm::expf(-value));
}

pub fn disigmoid(value: f32) -> f32{
	return value * (1.0 - value);
}

pub fn to_open(value: f32) -> f32{
	return value * 2.0 - 1.0;
}
#[derive(Serialize, Deserialize, Debug)]
pub struct Layer{
	layer_weights: DMatrix<f32>,
	layer_bias: DVector<f32>,
	layer_inputs: DVector<f32>,
	layer_outputs: DVector<f32>
}

pub fn create_layer(num_inputs: usize, num_nodes: usize) -> Layer {
	let mut rng = rand::thread_rng();
    let dist = rand::distributions::Standard;

	let mut layer = Layer { 
		layer_weights: DMatrix::from_distribution(num_nodes, num_inputs, &dist, &mut rng), 
		layer_bias: DVector::from_distribution(num_nodes, &dist, &mut rng),
		layer_inputs: DVector::zeros(num_inputs),
		layer_outputs: DVector::zeros(num_nodes) 
	};

	layer.layer_weights = layer.layer_weights.map(to_open);
	layer.layer_bias = layer.layer_bias.map(to_open);

	return layer;
}	

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNet{
    num_inputs: usize,
	num_outputs: usize,
	layers: Vec<(usize, Layer)>,
	accumelated_cost: f32,
	accumelated_guesses: DVector<f32>,
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
		let mut output = DVector::zeros(1);
		
		for layer in self.layers.iter_mut(){
			layer.1.layer_inputs += &input;
			
			output = &layer.1.layer_weights * &input;
			output += &layer.1.layer_bias;
			output = output.map(sigmoid);

			layer.1.layer_outputs += &output;
			input = output.clone();
		}

		return output;
	}

	pub fn calculate_cost(&mut self, previous_output: &DVector<f32>, correct_output: &DVector<f32>) -> f32{
		self.num_calcs += 1;
	
		
		let mut cost = 0.0;
		let delta = previous_output - correct_output;

		for n in 0..delta.data.len() {
			cost += libm::powf(delta[n], 2.0);
		}

		self.accumelated_cost += cost;
		self.accumelated_guesses += correct_output;

		return cost;
	}

	pub fn train(&mut self, learning_rate: f32) {
		let scale = 1.0 / self.num_calcs as f32;

		let avg_targets = &self.accumelated_guesses * scale;

		let avg_outputs = &self.layers[self.layers.len() - 1].1.layer_outputs * scale;
	
		let mut errors = (avg_outputs - avg_targets) * 2.0;
	
		for n in (0..self.layers.len()).rev(){
			let layer = &mut self.layers[n].1; 

			let avg_outputs = &layer.layer_outputs * scale;
			let mapped = avg_outputs.map(disigmoid);

			let mut gradients = DVector::identity(errors.data.len());
			for x in 0..mapped.data.len() {
				gradients[x] = mapped[x] * errors[x];
			}

			let avg_inputs = &layer.layer_inputs * scale;
			let transposed = avg_inputs.transpose();
			
			errors = layer.layer_weights.transpose() * &gradients;

			layer.layer_weights -= &gradients * transposed;
			layer.layer_bias -= &gradients * learning_rate;
			layer.layer_inputs *= 0.0;
			layer.layer_outputs *= 0.0;			
		}

		self.accumelated_guesses *= 0.0;
		self.accumelated_cost = 0.0;
		self.num_calcs = 0;
	}
}

pub fn create_neural_net(num_inputs: usize, num_outputs: usize, num_layers: usize, num_layer_nodes: usize) -> NeuralNet {
	let mut net = NeuralNet{
		num_inputs: num_inputs,
		num_outputs: num_outputs,
		layers: Vec::new(),
		accumelated_cost: 0.0,
		accumelated_guesses: DVector::zeros(num_outputs),
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

#[cfg(test)]
mod tests {
    use super::*;
	use nalgebra::dvector;
    #[test]
    fn init_test() {
        let net = create_neural_net(3,3,5,5);
        assert_eq!(net.num_inputs, 3);
        assert_eq!(net.num_outputs, 3);
        assert_eq!(net.layers.len(), 6);
        assert_eq!(net.accumelated_cost, 0.0);
        assert_eq!(net.accumelated_guesses, DVector::zeros(3));
        assert_eq!(net.num_calcs, 0);
    }

	#[test]
	fn feed_forward_test() {
		let mut net = create_neural_net(3,3,5,5);
		let input = DVector::identity(3);
		let outputs = net.feed_forward(&input);
		assert_eq!(outputs.len(), 3);
	}

	#[test]
	fn train_test() {
		let mut net = create_neural_net(3,3,5,5);
		let input = DVector::identity(3);
		let first: nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f32, nalgebra::Dyn, nalgebra::Const<1>>> = net.feed_forward(&input);
		
		let mut cost = 0.0;
		if first[0] > first[1] && first[0] > first[2]{
			cost += net.calculate_cost(&first, &dvector![1.0, 0.0, 0.0]);
		}
		else if first[1] > first[0] && first[1] > first[2]{
			cost += net.calculate_cost(&first, &dvector![0.0, 1.0, 0.0]);
		}
		else{
			cost += net.calculate_cost(&first, &dvector![0.0, 0.0, 1.0]);
		}		

		let learning_rate_max = 0.1;
		let learning_rate_min = 0.01;
		net.train(learning_rate_min + cost * (learning_rate_max - learning_rate_min));
	
		let second: nalgebra::Matrix<f32, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<f32, nalgebra::Dyn, nalgebra::Const<1>>> = net.feed_forward(&input);

		assert_ne!(first, second);
	}
}
