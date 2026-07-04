package pose.encoder;

import main.Main;
import pose.layer.ConvolutionalLayer;
import pose.layer.Layer;
import pose.tensor.Tensor;

public class PoseNet extends Layer
{
	private final EncoderBlock encoder1;
	private final EncoderBlock encoder2;
	private final EncoderBlock encoder3;

	private final DecoderBlock decoder1;
	private final DecoderBlock decoder2;
	private final DecoderBlock decoder3;

	private final ConvolutionalLayer convolution;
	
	private final float TIME = System.nanoTime();

	public PoseNet()
	{
		encoder1 = new EncoderBlock(3, 32);
		encoder2 = new EncoderBlock(32, 64);
		encoder3 = new EncoderBlock(64, 128);

		decoder1 = new DecoderBlock(128, 64, encoder3.getSkip());
		decoder2 = new DecoderBlock(64, 32, encoder2.getSkip());
		decoder3 = new DecoderBlock(32, 16, encoder1.getSkip());

		convolution = new ConvolutionalLayer(16, 15, 1, 1, 0);
	}

	@Override
	public Tensor forward(Tensor input)
	{
		System.out.println("Encoder 0 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");
		
		Tensor tensor1 = encoder1.forward(input);
		System.out.println("Encoder 1 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor2 = encoder2.forward(tensor1);
		System.out.println("Encoder 2 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor3 = encoder3.forward(tensor2);
		System.out.println("Encoder 3 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor4 = decoder1.forward(tensor3);
		System.out.println("Dencoder 1 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor5 = decoder2.forward(tensor4);
		System.out.println("Dencoder 2 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor6 = decoder3.forward(tensor5);
		System.out.println("Dencoder 3 forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor conv = convolution.forward(tensor6);
		System.out.println("Conv forward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		return conv;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor tensor0 = convolution.backward(gradient);
		System.out.println("tensor0 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor1 = decoder3.backward(tensor0);
		System.out.println("tensor1 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor2 = decoder2.backward(tensor1);
		System.out.println("tensor2 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor3 = decoder1.backward(tensor2);
		System.out.println("tensor3 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor4 = encoder3.backward(tensor3);
		System.out.println("tensor4 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor5 = encoder2.backward(tensor4);
		System.out.println("tensor5 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		Tensor tensor6 = encoder1.backward(tensor5);
		System.out.println("tensor6 Backward: " + (System.nanoTime() - TIME) / 1000000000f + " seconds");

		return tensor6;
	}
}
