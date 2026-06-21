package pose.encoder;

import pose.layer.ActivationLayer;
import pose.layer.ConvolutionalLayer;
import pose.layer.Layer;
import pose.layer.Sequential;
import pose.layer.SkipConnection;
import pose.layer.Tensor;
import pose.layer.UpSampleLayer;

public class DecoderBlock extends Layer
{
	private final Sequential sequential;
	private final SkipConnection skip;

	public DecoderBlock(int inC, int outC, SkipConnection skip)
	{
		this.skip = skip;

		sequential = new Sequential().add(new UpSampleLayer(2)).add(new ConvolutionalLayer(inC + outC, outC, 3, 1, 1)).add(new ActivationLayer());

	}

	@Override
	public Tensor forward(Tensor input)
	{
		Tensor skipTensor = skip.getEncoderOutput();
		Tensor merged = concat(input, skipTensor);
		return sequential.forward(merged);
	}

	private Tensor concat(Tensor input, Tensor skipTensor)
	{
		if (input.getBatchSize() != skipTensor.getBatchSize() || input.getHeight() != skipTensor.getHeight() || input.getWidth() != skipTensor.getWidth())
		{
			throw new IllegalArgumentException("Cannot concatenate tensors with different spatial dimensions");
		}
		int inChannels = input.getChannels();
		int skipChannels = skipTensor.getChannels();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();
		int skipHeight = skipTensor.getHeight();
		int skipWidth = skipTensor.getWidth();
		
		int outputChannels = inChannels + skipChannels;
		Tensor output = new Tensor(input.getBatchSize(), outputChannels, inHeight, inWidth);
		
		for (int i = 0; i < output.getBatchSize(); i++)
		{
			for (int h = 0; h < inChannels; h++)
			{
				for (int y = 0; y < inHeight; y++)
				{
					for (int x = 0; x < inWidth; x++)
					{
						output.set(i, h, y, x, input.get(i, h, y, x));
					}
				}
			}
			for (int h = 0; h < skipChannels; h++)
			{
				for (int y = 0; y < skipHeight; y++)
				{
					for (int x = 0; x < skipWidth; x++)
					{
						output.set(i, h + inChannels, y, x, skipTensor.get(i, h, y, x));
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
