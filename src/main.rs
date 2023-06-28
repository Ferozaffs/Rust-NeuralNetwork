pub mod neural_nets;
use nalgebra::DVector;


fn main() {
    let mut net = neural_nets::create_neural_net(3,3,5,5);
    net.debug_print();

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Standard;

    let mut previous_avg_cost: f32 = 1.0;
	for _generation in 0..5 {
		let mut cost: f32 = 0.0;

		for _test in 0..1 {
            let input = DVector::from_distribution(3, &dist, &mut rng);
            let outputs = net.feed_forward(&input);
            println!("INPUT{} OUTPUT{}",&input, &outputs);
        }
    }
}
