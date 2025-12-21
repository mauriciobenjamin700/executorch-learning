# APrendendo a Usar ExecuTorch

O ExecuTorch é a solução do PyTorch para treinamento e inferência no Edge, fornecendo portabilidade, produtividade e desempenho para plataformas de computação de borda.

## Começando com o ExecuTorch

O seguinte é necessário para instalar as bibliotecas de host do ExecuTorch, necessárias para exportar modelos e executar a partir do Python. Os requisitos para dispositivos de usuário final alvo são dependentes do backend.

- Python 3.10 - 3.12
- g++ versão 7 ou superior, clang++ versão 5 ou superior, ou outra cadeia de ferramentas compatível com C++17.
- Linux (x86_64 ou ARM64), macOS (ARM64) ou Windows (x86_64).
- Os sistemas macOS baseados na Intel exigem a construção do PyTorch a partir da fonte (consulte para obter instruções).
- No Windows, Visual Studio 2022 ou posterior.

### Instalação

```bash
pip install executorch
```

### Preparando o Modelo

Exportar é o processo de pegar um modelo do PyTorch e convertê-lo para o formato de arquivo `.pte` usado pelo tempo de execução do ExecuTorch. Isso é feito usando APIs Python. Arquivos PTE para modelos comuns, como o Llama 3.2, podem ser encontrados no HuggingFace sob a [Comunidade ExecuTorch](https://huggingface.co/executorch-community). Esses modelos foram exportados e rebaixados para o ExecuTorch, e podem ser implantados diretamente sem a necessidade de passar pelo processo de redução.

### Selecionando um Backend

O ExecuTorch fornece aceleração de hardware para uma ampla variedade de hardware. Os backends mais comumente usados são XNNPACK, para Arm e x86 CPU, Core ML (para iOS), Vulkan (para GPUs Android) e Qualcomm (para telefones Android com alimentação Qualcomm).

Para casos de uso móveis, considere o uso do XNNPACK para Android e Core ML ou XNNPACK para iOS como um primeiro passo. Consulte [Backends de hardware](https://docs.pytorch.org/executorch/stable/backends-overview.html) para obter mais informações.  

Os backends do ExecuTorch fornecem aceleração de hardware para alvos de hardware específicos, permitindo que os modelos sejam executados de forma eficiente em dispositivos que vão desde telefones celulares até sistemas embarcados e DSPs. Durante o processo de exportação e redução, o ExecuTorch otimiza seu modelo para o backend escolhido, resultando em um `.pte` arquivo especializado para esse hardware. Para suportar várias plataformas (por exemplo, Core ML no iOS, Arm CPU no Android), você normalmente gera um dedicado `.pt` earquivo para cada backend.

A escolha do backend é informada pelo hardware em que seu modelo será executado. Cada backend tem seus próprios requisitos de hardware e nível de suporte a modelo/operador. Consulte a documentação para cada backend para detalhes.

Com o arquivo `.pte`, o ExecuTorch identifica partições de modelo suportadas pelo backend. Estes são processados com antecedência para uma execução eficiente. Os operadores não suportados pelo delegado são executados usando o fallback portátil da CPU (por exemplo, XNNPACK), permitindo a aceleração parcial. Você também pode especificar vários particionadores em ordem de prioridade, para que as operações de GPU não suportadas possam cair de volta para a CPU, por exemplo.

Backends são a ponte entre o seu modelo exportado e o hardware em que ele é executado. A escolha do back-end certo garante que seu modelo aproveite ao máximo a aceleração específica do dispositivo, o desempenho de balanceamento, a compatibilidade e o uso de recursos.

### Exportando

A exportação é feita usando APIs Python. O ExecuTorch fornece um alto grau de personalização durante o processo de exportação, mas o fluxo típico é o seguinte. Este exemplo usa a implementação do modelo de classificação de imagem MobileNet V2 na torchvision, mas o processo suporta qualquer modelo PyTorch compatível com a exportação. Para os modelos de Hugging Face, Você pode encontrar uma lista de modelos suportados no repo [huggingface/optimum-executorch](https://github.com/huggingface/optimum-executorch).

```python
import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
```

Se o modelo exigir tamanhos de entrada variados, você precisará especificar as diferentes dimensões e limites como parte do `export` chamada. Consulte [Modelo Exportar e Rebaixar](https://docs.pytorch.org/executorch/stable/using-executorch-export.html) para mais informações.

O backend de hardware para o alvo é controlado pelo parâmetro partiker para `to_edge_transform_and_lower`. Neste exemplo, o `XnnpackPartitioner` é usado para segmentar CPUs móveis. Consulte a [documentação específica](https://docs.pytorch.org/executorch/stable/backends-overview.html) para obter informações sobre como usar cada back-end.

A quantização também pode ser feita neste estágio para reduzir o tamanho do modelo e o tempo de execução. A quantização é específica do backend. Consulte a documentação para o backend de destino para obter uma descrição completa dos esquemas de quantização suportados.

### Testando o Modelo

Depois de gerar com sucesso um arquivo .pte, é comum usar as APIs de tempo de execução do Python para validar o modelo na plataforma de desenvolvimento. Isso pode ser usado para avaliar a precisão do modelo antes de executar no dispositivo.

Para o modelo MobileNet V2 da torchvision usado neste exemplo, as entradas de imagem são esperadas como um tensor flutuante32 normalizado com umas dimensões de (lote, canais, altura, largura). A saída é um tensor contendo logits de classe. Consulte [torchvision.models.mobilenet_v2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html) para obter mais informações sobre o formato de tensor de entrada e saída para este modelo.

```python
import torch
from executorch.runtime import Runtime
from typing import List

runtime = Runtime.get()

input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
program = runtime.load_program("model.pte")
method = program.load_method("forward")
output: List[torch.Tensor] = method.execute([input_tensor])
print("Run successfully via executorch")

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torchvision.models as models

eager_reference_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
eager_reference_output = eager_reference_model(input_tensor)

print("Comparing against original PyTorch module")
print(torch.allclose(output[0], eager_reference_output, rtol=1e-3, atol=1e-5))
```

Para exemplos completos de exportação e execução do modelo, consulte o [repositório do GitHub de exemplos](https://github.com/meta-pytorch/executorch-examples/tree/main/mv2/python).

Além disso, para os modelos Hugging Face, a biblioteca [huggingface/bptimum-executorch](https://github.com/huggingface/optimum-executorch) simplifica a execução desses modelos de ponta a ponta com o ExecuTorch usando APIs de Hugging Face familiares. Visite o repositório para exemplos específicos e modelos suportados.

### Pacotes Para Facilitar A Implantação Mobile

- [ExecuTorch Flutter](https://pub.dev/packages/executorch_flutter)
- [React Native ExecuTorch](https://github.com/software-mansion/react-native-executorch)

## Referências

- [Introduction](https://docs.pytorch.org/executorch/stable/intro-section.html)
- [Getting Start](https://docs.pytorch.org/executorch/stable/getting-started.html)
- [ExecuTorch Backends](https://docs.pytorch.org/executorch/stable/backends-overview.html)
- [ExecuTorch Export](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)
