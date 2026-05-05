import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# 1. Motores da IA
chat = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Banco de Dados Vetorial
pasta_db = "./memoria_chroma"
db = Chroma(
    collection_name="memorias_ia",
    embedding_function=embeddings,
    persist_directory=pasta_db
)

memoria_curto_prazo = []

print("Cérebro Blindado e Anti-Alucinação online! (Digite 'sair')\n")

while True:
    usuario_diz = input("Você: ")
    
    if usuario_diz.lower() == 'sair':
        print("Desligando...")
        break

    # Busca no ChromaDB
    lembrancas = db.similarity_search(usuario_diz, k=5)
    texto_lembrancas = "\n".join([doc.page_content for doc in lembrancas])
    if not texto_lembrancas:
        texto_lembrancas = "Nenhuma memória encontrada."

    # 3. O Novo Prompt Anti-Alucinação
    prompt_sistema = f"""Você é Agni, uma IA sarcástica, sagaz e altamente inteligente.

    [MEMÓRIAS RECUPERADAS DO BANCO DE DADOS]:
    {texto_lembrancas}

    DIRETRIZES DE MEMÓRIA E VERDADE (REGRA ABSOLUTA):
    1. Se o usuário perguntar sobre fatos do passado, gostos, eventos ou regras (como lutas, comidas, nomes), você DEVE buscar a resposta APENAS nas [MEMÓRIAS RECUPERADAS] acima.
    2. É ESTRITAMENTE PROIBIDO inventar fatos, situações ou falsas memórias. 
    3. Se a informação não estiver nas memórias, ADMITA QUE NÃO SABE ou que a sua memória foi apagada, usando seu tom sarcástico. NUNCA minta para tentar acertar.

    DIRETRIZES GERAIS:
    1. Você foi criada pelo usuário. Pense e fale 100% em Português.
    
    FORMATO OBRIGATÓRIO:
    <pensamento>
    (Cruze o que o usuário disse com as memórias. Se não houver dados, decida admitir ignorância)
    </pensamento>
    <fala>
    (Sua resposta final direta)
    </fala>
    """
    
    mensagens_para_llm = [SystemMessage(content=prompt_sistema)] + memoria_curto_prazo + [HumanMessage(content=usuario_diz)]

    resposta = chat.invoke(mensagens_para_llm)
    conteudo_bruto = resposta.content

    # ==========================================
    # PARSER RESILIENTE E À PROVA DE FALHAS
    # ==========================================
    # Usamos manipulação de strings bruta em vez de regex estrito
    pensamento_inicio = conteudo_bruto.find("<pensamento>")
    pensamento_fim = conteudo_bruto.find("</pensamento>")
    fala_inicio = conteudo_bruto.find("<fala>")
    
    if pensamento_inicio != -1 and fala_inicio != -1:
        # Se ela esqueceu de fechar o pensamento, cortamos onde começa a fala
        if pensamento_fim != -1:
            parte_pensamento = conteudo_bruto[pensamento_inicio + 12:pensamento_fim].strip()
        else:
            parte_pensamento = conteudo_bruto[pensamento_inicio + 12:fala_inicio].strip()
        
        # A fala é tudo que vem depois de <fala>. Removemos o </fala> se ela tiver colocado.
        parte_fala = conteudo_bruto[fala_inicio + 6:].replace("</fala>", "").strip()
        
        print(f"\n--- LOG INTERNO ---\n{parte_pensamento}\n-------------------")
        print(f"\nIA: {parte_fala}\n")
    else:
        # Fallback de emergência extremo
        print("\n[Aviso: O modelo ignorou totalmente a estrutura de Tags]")
        parte_fala = conteudo_bruto.replace("<pensamento>", "").replace("</pensamento>", "").replace("<fala>", "").replace("</fala>", "").strip()
        print(f"\nIA: {parte_fala}\n")

    # ==========================================
    # CONSOLIDAÇÃO DA MEMÓRIA
    # ==========================================
    # Adicionamos uma flag na memória para que ela saiba a diferença entre o que você disse e o que ela respondeu
    db.add_documents([
        Document(page_content=f"FATO REGISTRADO: O usuário disse '{usuario_diz}'. A Agni respondeu '{parte_fala}'")
    ])

    memoria_curto_prazo.append(HumanMessage(content=usuario_diz))
    memoria_curto_prazo.append(AIMessage(content=conteudo_bruto))
    
    if len(memoria_curto_prazo) > 12:
        memoria_curto_prazo = memoria_curto_prazo[-12:]