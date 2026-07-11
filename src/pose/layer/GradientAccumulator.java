package pose.layer;

public class GradientAccumulator
{
	private float[]gradientInput;
	private float[]weightGradient;
	private float[]bias;
	
	public GradientAccumulator(float[] gradientInput, float[] weightGradient, float[] bias)
	{
		this.gradientInput = gradientInput;
		this.weightGradient = weightGradient;
		this.bias = bias;
	}

	public float[] getGradientInput()
	{
		return gradientInput;
	}

	public float[] getWeightGradient()
	{
		return weightGradient;
	}

	public float[] getBias()
	{
		return bias;
	}
}
