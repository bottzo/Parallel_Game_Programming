struct PixelShaderInput
{
    float4 Color: COLOR;
    float2 uvs  : UV;
};

Texture2D DiffuseTexture      : register(t0);
SamplerState StaticSampler    : register(s0);

float4 main(PixelShaderInput IN) : SV_Target
{
    float4 texColor = DiffuseTexture.Sample(StaticSampler, IN.uvs);
    return lerp(texColor, IN.Color, 0.5);
}