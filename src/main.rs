pub mod neural_nets;
use nalgebra::{DVector, dvector};
use neural_nets::NeuralNet;
use std::io::{self, Read, Write};
use std::fs::File;

use std::env;
fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    
    let stdin = io::stdin();
    let mut filename = String::new();

    print!("Load trained network or keep empty to train:");
    io::stdout().flush().unwrap();
    stdin.read_line(&mut filename).expect("error: unable to read user input");
    
    let mut net: NeuralNet;
    if filename.len() > 1 {
        let mut file = File::open(&filename.trim()).expect("Error opening File");
        let mut data = String::new();
        file.read_to_string(&mut data).expect("Unable to read file");
        net = serde_json::from_str(&data).unwrap();
    }
    else {
        net = neural_nets::create_neural_net(3,3,5,5);
    
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Standard;
        
        let num_generations = 50000;
        let num_tests = 100;
        
        for _generation in 0..num_generations {
            let mut cost: f32 = 0.0;
            
            for _test in 0..num_tests {
                let input = DVector::from_distribution(3, &dist, &mut rng);
                let outputs = net.feed_forward(&input);
                
                cost += get_cost(&mut net, outputs);	
            }
            
            let learning_rate_max = 0.1;
            let learning_rate_min = 0.01;
            let avg_cost = cost / num_tests as f32;
            net.train(learning_rate_min + avg_cost * (learning_rate_max - learning_rate_min));
            
            if _generation % 100 == 0 {
                println!("{} Avg.cost", avg_cost);
            }
        }
        
        let output_filename = "trained_network.json";
        std::fs::write(output_filename ,serde_json::to_string(&net).unwrap()).unwrap();
        
        print!("\n\nSaved training to file: {}\n\n", output_filename);    
    }
 
    let mut r = String::new();
    let mut g = String::new();
    let mut b = String::new();
    
    println!("Test result by specifing color by channel.");
    print!("RED: ");
    io::stdout().flush().unwrap();
    stdin.read_line(&mut r).expect("error: unable to read user input");
    print!("GREEN: ");
    io::stdout().flush().unwrap();
    stdin.read_line(&mut g).expect("error: unable to read user input");
    print!("BLUE: ");
    io::stdout().flush().unwrap();
    stdin.read_line(&mut b).expect("error: unable to read user input");

    let input = &dvector![
        r.trim().parse::<f32>().unwrap(),
        g.trim().parse::<f32>().unwrap(),
        b.trim().parse::<f32>().unwrap()
    ];
    let outputs = net.feed_forward(input);

    let major_color: String;
    if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        major_color = "RED".to_string();
    }
    else if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        major_color = "GREEN".to_string();
    }
    else {
        major_color = "BLUE".to_string();
    }

    println!("\n\nOutputs: {}", outputs);
    println!("Network think color is mostly: {}", major_color);

}

pub fn get_cost(neural_net: &mut NeuralNet, outputs: DVector<f32>) -> f32 {
    if outputs[0] > outputs[1] && outputs[0] > outputs[2]{
        return neural_net.calculate_cost(&outputs, &dvector![1.0, 0.0, 0.0]);
    }
    else if outputs[1] > outputs[0] && outputs[1] > outputs[2]{
        return neural_net.calculate_cost(&outputs, &dvector![0.0, 1.0, 0.0]);
    }
    else{
        return neural_net.calculate_cost(&outputs, &dvector![0.0, 0.0, 1.0]);
    }	
}

