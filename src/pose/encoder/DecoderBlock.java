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
	private final UpSampleLayer upsample;
	private final SkipConnection skip;

	private int decoderChannels;
	private int skipChannels;

	public DecoderBlock(int inC, int outC, SkipConnection skip)
	{
		this.skip = skip;

		upsample = new UpSampleLayer(2);

		sequential = new Sequential().add(new ConvolutionalLayer(inC + outC, outC, 3, 1, 1)).add(new ActivationLayer()).add(new ConvolutionalLayer(outC, outC, 3, 1, 1)).add(new ActivationLayer());
	}

	@Override
	public Tensor forward(Tensor input)
	{
		Tensor up = upsample.forward(input);
		Tensor skipTensor = skip.getEncoderOutput();

		decoderChannels = up.getChannels();
		skipChannels = skipTensor.getChannels();

		Tensor merged = concat(up, skipTensor);
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
		Tensor mergedGradient = sequential.backward(gradient);

		Tensor decoderGradient = new Tensor(mergedGradient.getBatchSize(), decoderChannels, mergedGradient.getHeight(), mergedGradient.getWidth());
		Tensor skipGradient = new Tensor(mergedGradient.getBatchSize(), skipChannels, mergedGradient.getHeight(), mergedGradient.getWidth());

		for (int b = 0; b < mergedGradient.getBatchSize(); b++)
		{
			for (int c = 0; c < decoderChannels; c++)
			{
				for (int y = 0; y < mergedGradient.getHeight(); y++)
				{
					for (int x = 0; x < mergedGradient.getWidth(); x++)
					{
						decoderGradient.set(b, c, y, x, mergedGradient.get(b, c, y, x));
					}
				}
			}
			for (int c = 0; c < skipChannels; c++)
			{
				for (int y = 0; y < mergedGradient.getHeight(); y++)
				{
					for (int x = 0; x < mergedGradient.getWidth(); x++)
					{
						skipGradient.set(b, c, y, x, mergedGradient.get(b, c + decoderChannels, y, x));
					}
				}
			}
		}
		skip.setGradient(skipGradient);
		
		return upsample.backward(decoderGradient);
	}
}
