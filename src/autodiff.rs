use dfdx::{
    shapes::{Rank0, Rank2},
    tensor::{AsArray, AutoDevice, Gradients, NoneTape, OwnedTape, SampleTensor, Tensor, Trace},
    tensor_ops::{Backward, MeanTo, TryMatMul},
};

#[cfg(test)]
mod autodiff_tests {
    use dfdx::prelude::*;

    #[test]
    fn test_dfdx_grads() {
        println!("Testing dfdx package!");

        let dev = AutoDevice::default();

        // Sample random tensors
        let weight: Tensor<Rank2<4, 2>, f32, _, NoneTape> = dev.sample_normal();
        let a: Tensor<Rank2<3, 4>, f32, _, NoneTape> = dev.sample_normal();

        // Allocate gradients for tracking
        let grads = Gradients::leaky();

        // Trace tensor 'a' for gradient tracking
        let b: Tensor<Rank2<3, 4>, _, _, OwnedTape<f32, AutoDevice>> = a.trace(grads);

        // Forward computation
        let c: Tensor<Rank2<3, 2>, _, _, OwnedTape<f32, AutoDevice>> = b.matmul(weight.clone());
        let d: Tensor<Rank2<3, 2>, _, _, OwnedTape<f32, AutoDevice>> = c.sin();
        let e: Tensor<Rank0, _, _, OwnedTape<f32, AutoDevice>> = d.mean();

        // Backward pass to compute gradients
        let gradients: Gradients<f32, AutoDevice> = e.backward();

        // Extract gradients for 'a' and 'weight'
        let a_grad: [[f32; 4]; 3] = gradients.get(&a).array();
        let weight_grad: [[f32; 2]; 4] = gradients.get(&weight).array();

        println!("{:?}", a_grad);
        println!("{:?}", weight_grad);

        // Simple checks: gradients should be finite numbers
        for row in &a_grad {
            for v in row {
                assert!(v.is_finite(), "Non-finite gradient in a_grad");
            }
        }
        for row in &weight_grad {
            for v in row {
                assert!(v.is_finite(), "Non-finite gradient in weight_grad");
            }
        }
    }
}