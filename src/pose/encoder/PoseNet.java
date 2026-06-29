package pose.encoder;

import pose.layer.ConvolutionalLayer;
import pose.layer.Layer;
import pose.layer.Tensor;

public class PoseNet extends Layer
{
	private final EncoderBlock encoder1;
	private final EncoderBlock encoder2;
	private final EncoderBlock encoder3;
	
	private final DecoderBlock decoder1;
	private final DecoderBlock decoder2;
	private final DecoderBlock decoder3;
	
	private final ConvolutionalLayer convolution;
	
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
		Tensor tensor1 = encoder1.forward(input);
		Tensor tensor2 = encoder2.forward(tensor1);
		Tensor tensor3 = encoder3.forward(tensor2);
		
		Tensor tensor4 = decoder1.forward(tensor3);
		Tensor tensor5 = decoder2.forward(tensor4);
		Tensor tensor6 = decoder3.forward(tensor5);
		
		return convolution.forward(tensor6);
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor tensor0 = convolution.backward(gradient);
		Tensor tensor1 = decoder3.backward(tensor0);
		Tensor tensor2 = decoder2.backward(tensor1);
		Tensor tensor3 = decoder1.backward(tensor2);
		
		Tensor tensor4 = encoder3.backward(tensor3);
		Tensor tensor5 = encoder2.backward(tensor4);
		Tensor tensor6 = encoder1.backward(tensor5);
		
		return tensor6;
	}
}
