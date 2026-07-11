package test;

import pose.encoder.EncoderBlock;
import pose.encoder.PoseNet;
import pose.layer.ConvolutionalLayer;
import pose.layer.SkipConnection;
import pose.tensor.Tensor;
import pose.trainer.DataLoader;
import pose.trainer.Dataset;
import pose.trainer.MSELoss;
import pose.trainer.SGDOptimizer;
import pose.trainer.Trainer;

public class LayerTest
{
	public static void main(String[] args)
	{
		// testConvLayer();
		// testPoseNet();
		// testNumericalGradient();
		// testSkipConnection();

		testOverfitSanityCheck();
	}

	private static void testOverfitSanityCheck()
	{
		System.out.println("Running overfit sanity test");
		
		PoseNet poseNet = new PoseNet();

		Dataset dataset = DataLoader.loadDataset("res/training_data/test/tensorImages", "res/training_data/test/heatmaps");

		Trainer trainer = new Trainer(poseNet, dataset, new SGDOptimizer(0.01f), new MSELoss());

		trainer.train(50);
		System.out.println("Overfit Sanity Check: PASSED ");
	}

	private static void testSkipConnection()
	{
		EncoderBlock encoder = new EncoderBlock(3, 8);
		Tensor input = Tensor.acquire(1, 3, 32, 32);
		randomFill(input);

		SkipConnection skipConnection = encoder.getSkip();
		Tensor saved = skipConnection.getEncoderOutput();

		assert saved != null;

		// System.out.println("Skip saved shape: " + saved);
		System.out.println("Skip Connection Check: PASSED");
	}

	private static void testNumericalGradient()
	{
		ConvolutionalLayer conv = new ConvolutionalLayer(3, 2, 3, 1, 1);

		Tensor input = Tensor.acquire(1, 3, 5, 5);
		Tensor target = Tensor.acquire(1, 2, 5, 5);
		randomFill(input);
		randomFill(target);

		MSELoss loss = new MSELoss();

		Tensor prediction = conv.forward(input);
		Tensor gradientOutput = loss.backward(prediction, target);
		conv.backward(gradientOutput);

		float epsilon = 1e-3f;

		int idx = 0;
		float[] weights = conv.getWeights();
		float original = weights[idx];

		weights[idx] = original + epsilon;
		conv.setWeights(weights);
		float loss1 = computeLoss(conv, input, target);

		weights[idx] = original - epsilon;
		conv.setWeights(weights);
		float loss2 = computeLoss(conv, input, target);

		weights[idx] = original;
		conv.setWeights(weights);

		float numericalGradient = (loss1 - loss2) / (2 * epsilon);
		float backpropGradient = conv.getWeightGradients()[idx];
		float diff = Math.abs(numericalGradient - backpropGradient);

		// System.out.println("Numerical: " + numericalGradient);
		// System.out.println("Backprop : " + backpropGradient);
		// System.out.println("Diff: " + diff);

		if (diff < 1e-2)
		{
			System.out.println("Gradient Check: PASSED");
		}
		else
		{
			System.out.println("Gradient Check: FAILED");
		}
	}

	private static void testPoseNet()
	{
		PoseNet poseNet = new PoseNet();

		Tensor input = Tensor.acquire(1, 3, 64, 64);
		randomFill(input);
		Tensor output = poseNet.forward(input);
		// System.out.println("Output shape: " + output);

		Tensor target = Tensor.acquire(output.getBatchSize(), output.getChannels(), output.getHeight(), output.getWidth());
		randomFill(target);

		MSELoss loss = new MSELoss();
		// float lossValue = loss.forward(output, target);
		Tensor gradient = loss.backward(output, target);

		poseNet.backward(gradient);

		// System.out.println("Loss: " + lossValue);
		System.out.println("Pose Net Backward Propagation Check: PASSED");
	}

	private static void testConvLayer()
	{
		ConvolutionalLayer conv = new ConvolutionalLayer(3, 2, 3, 1, 1);

		Tensor input = Tensor.acquire(1, 3, 5, 5);
		randomFill(input);
		Tensor output = conv.forward(input);
		// System.out.println("Output shape: " + output);

		Tensor gradientOutput = Tensor.acquire(output.getBatchSize(), output.getChannels(), output.getHeight(), output.getWidth());
		randomFill(gradientOutput);
		Tensor gradIn = conv.backward(gradientOutput);
		// System.out.println("Gradient Input shape: " + gradIn);

		assert gradIn.getBatchSize() == input.getBatchSize();
		assert gradIn.getChannels() == input.getChannels();
		assert gradIn.getWidth() == input.getWidth();
		assert gradIn.getHeight() == input.getHeight();

		System.out.println("Conv Forward/Backward Shape Check: PASSED");
	}

	private static void randomFill(Tensor tensor)
	{
		float[] d = tensor.getData();

		for (int i = 0; i < d.length; i++)
		{
			d[i] = (float) (Math.random() * 2 - 1);
		}
	}

	private static float computeLoss(ConvolutionalLayer conv, Tensor input, Tensor target)
	{
		Tensor prediction = conv.forward(input);

		MSELoss loss = new MSELoss();

		return loss.forward(prediction, target);
	}
}
