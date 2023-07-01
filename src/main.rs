pub mod neural_nets;
use nalgebra::{DVector, dvector};

use std::env;
fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let mut net = neural_nets::create_neural_net(3,3,5,5);
    net.debug_print();

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Standard;

    let num_generations = 500000;
    let num_tests = 100;

	for _generation in 0..num_generations {
		let mut cost: f32 = 0.0;

		for _test in 0..num_tests {
            let input = DVector::from_distribution(3, &dist, &mut rng);
            let outputs = net.feed_forward(&input);

            if outputs[0] > outputs[1] && outputs[0] > outputs[2]{
				cost += net.calculate_cost(&outputs, &dvector![1.0, 0.0, 0.0]);
			}
			else if outputs[1] > outputs[0] && outputs[1] > outputs[2]{
				cost += net.calculate_cost(&outputs, &dvector![0.0, 1.0, 0.0]);
			}
			else{
				cost += net.calculate_cost(&outputs, &dvector![0.0, 0.0, 1.0]);
			}		
        }

		let learning_rate_max = 0.1;
		let learning_rate_min = 0.01;
        let avg_cost = cost / num_tests as f32;
		net.train(learning_rate_min + avg_cost * (learning_rate_max - learning_rate_min));
	
        println!("{} Avg.cost\n", avg_cost);
    }
}

