package pose.layer;

import java.util.Arrays;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class ConvolutionalLayer extends Layer
{
	private final int inChannels;
	private final int outChannels;
	private final int kernalSize;
	private final int stride;
	private final int padding;

	private final float[] weights;
	private final float[] bias;
	private float[] weightGradients;
	private float[] biasGradients;

	public ConvolutionalLayer(int inChannels, int outChannels, int kernalSize, int stride, int padding)
	{
		this.inChannels = inChannels;
		this.outChannels = outChannels;
		this.kernalSize = kernalSize;
		this.stride = stride;
		this.padding = padding;

		int weightCount = outChannels * inChannels * kernalSize * kernalSize;
		this.weights = new float[weightCount];
		this.bias = new float[outChannels];
		this.weightGradients = new float[weights.length];
		this.biasGradients = new float[bias.length];

		// Uniform Weight Distribution
		int fanIn = inChannels * kernalSize * kernalSize;
		double limit = Math.sqrt(6 / fanIn);
		uniform(weights, limit);
	}

	@Override
	public Tensor forward(Tensor input)
	{
		this.input = input;

		int batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = Math.floorDiv(inHeight + (padding * 2) - kernalSize, stride) + 1;
		int outWidth = Math.floorDiv(inWidth + (padding * 2) - kernalSize, stride) + 1;

		Tensor output = new Tensor(batchSize, outWidth, outHeight, outChannels);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int chout = 0; chout < outChannels; chout++)
			{
				for (int outH = 0; outH < outHeight; outH++)
				{
					for (int outW = 0; outW < outWidth; outW++)
					{
						float sum = bias[chout];

						for (int chin = 0; chin < inChannels; chin++)
						{
							for (int kernalX = 0; kernalX < kernalSize; kernalX++)
							{
								for (int kernalY = 0; kernalY < kernalSize; kernalY++)
								{
									int inX = outW * stride + kernalX - padding;
									int inY = outH * stride + kernalY - padding;

									if (inY >= 0 && inX >= 0 && inY < inHeight && inX < inWidth)
									{
										int dataIndex = inY * (inWidth * inChannels) + inX * inChannels + chin;
										int weightIndex = chout * (kernalSize * kernalSize * inChannels) + kernalY * (kernalSize * inChannels) + kernalX * inChannels + chin;

										sum += input.getData()[dataIndex] * weights[weightIndex];
									}
								}
							}
						}
						output.set(batch, outH, outW, chout, sum);
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradientInput = new Tensor(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());

		Arrays.fill(weightGradients, 0f);
		Arrays.fill(biasGradients, 0f);

		int batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = gradient.getHeight();
		int outWidth = gradient.getWidth();

		float[] inputData = input.getData();
		float[] gradData = gradient.getData();
		float[] gradInputData = gradientInput.getData();

		for (int batch = 0; batch < gradient.getBatchSize(); batch++)
		{
			for (int outY = 0; outY < gradient.getHeight(); outY++)
			{
				for (int outX = 0; outX < gradient.getWidth(); outX++)
				{
					for (int chout = 0; chout < gradient.getChannels(); chout++)
					{
						int gradIndex = ((batch * outHeight + outY) * outWidth + outX) * outChannels + chout;
						float grad = gradData[gradIndex];
						
						biasGradients[chout] += grad;

						for (int kernalY = 0; kernalY < kernalSize; kernalY++)
						{
							for (int kernalX = 0; kernalX < kernalSize; kernalX++)
							{
								int inY = outY * stride + kernalY - padding;
								int inX = outX * stride + kernalX - padding;

								if (inY < 0 || inX < 0 || inY >= inHeight || inX >= inWidth)
								{
									continue;
								}
								for (int chin = 0; chin < inChannels; chin++)
								{
									int inputIndex = ((batch * inHeight + inY) * inWidth + inX) * inChannels + chin;
									int weightIndex = ((chout * inChannels + chin) * kernalSize + kernalY) * kernalSize + kernalX;

									float inputValue = inputData[inputIndex];
									float weightValue = weightGradients[weightIndex];

									weightGradients[weightIndex] += grad * inputValue;
									gradInputData[inputIndex] += grad * weightValue;
								}
							}
						}
					}
				}
			}
		}
		return gradientInput;
	}
	
	private static void uniform(float[]array, double limit)
	{
		Random random = new Random();
		
		for(int i=0; i<array.length; i++)
		{
			array[i] = (float)((random.nextDouble() * 2 - 1) * limit);
		}
	}
}
