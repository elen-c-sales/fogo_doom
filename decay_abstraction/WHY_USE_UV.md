# Por que usar o uv?

O **uv** é uma ferramenta moderna e extremamente rápida para gerenciar projetos Python. Ele substitui ferramentas antigas como pip, poetry e virtualenv, cuidando de tudo (desde a instalação da versão correta do Python até o gerenciamento das bibliotecas) de uma forma muito mais ágil e eficiente.

Pense nele como um "super-gerente" para seus projetos Python, feito para economizar seu tempo.

## Como instalar

A instalação é super simples. No seu terminal (macOS ou Linux), basta rodar:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Se precisar de mais detalhes ou estiver no Windows, confira a [documentação oficial](https://github.com/astral-sh/uv).

## Como rodar o script de comparação

Uma das melhores coisas do uv é que você não precisa criar e ativar ambientes virtuais manualmente toda hora. O uv faz isso para você.

Para rodar o script de comparação de abordagens (`compare_approaches.py`), basta estar na raiz do projeto e executar:

```bash
uv run decay_abstraction/compare_approaches.py
```

O uv vai verificar se você tem as dependências necessárias instaladas e, se não tiver, vai baixar e configurar tudo automaticamente antes de rodar o script.

## Arquivos importantes

Para que essa mágica aconteça, o uv utiliza alguns arquivos que você encontrará na raiz do projeto:

*   **`.python-version`**: Diz ao uv exatamente qual versão do Python deve ser usada neste projeto.
*   **`pyproject.toml`**: É onde listamos todas as bibliotecas que o projeto precisa para funcionar (como o `numpy` ou `matplotlib`).
*   **`uv.lock`**: É um arquivo gerado automaticamente que "tranca" as versões exatas de todas as dependências. Isso garante que o projeto rode exatamente da mesma forma no seu computador e no de qualquer outra pessoa.
