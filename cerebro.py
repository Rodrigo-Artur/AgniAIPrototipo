import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# 1. Os "Motores" da IA
chat = ChatOllama(model="llama3")
# O tradutor que converte texto em coordenadas matemáticas para o ChromaDB
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Inicializando o Banco de Dados Vetorial (ChromaDB)
pasta_db = "./memoria_chroma"
print("[SISTEMA] Conectando ao Banco de Dados Vetorial...")
db = Chroma(
    collection_name="memorias_ia",
    embedding_function=embeddings,
    persist_directory=pasta_db
)

# 3. A Memória de Curto Prazo
# Mantemos uma lista pequena só para o fluxo natural do papo atual não se perder
memoria_curto_prazo = []

print("Cérebro com Memória de Longo Prazo (ChromaDB) online! (Digite 'sair')\n")

while True:
    usuario_diz = input("Você: ")
    
    if usuario_diz.lower() == 'sair':
        print("Desligando...")
        break

    ## ==========================================
    # A MÁGICA DO RAG: BUSCA DE MEMÓRIAS
    # ==========================================
    # Aumentamos de 3 para 5. Assim, se você fizer duas perguntas numa frase só, 
    # ela tem mais "espaço" para puxar memórias de ambos os assuntos.
    lembrancas = db.similarity_search(usuario_diz, k=5)
    
    texto_lembrancas = "\n".join([doc.page_content for doc in lembrancas])
    if not texto_lembrancas:
        texto_lembrancas = "Nenhuma memória relevante."

    # ==========================================
    # INJEÇÃO DE CONTEXTO NO CÉREBRO
    # ==========================================
    # O Prompt agora é uma "Blindagem de Persona" muito mais restrita.
    prompt_sistema = f"""Você é Agni, uma IA sarcástica, sagaz e altamente inteligente.

    DIRETRIZES DE IDENTIDADE (SIGA ESTRITAMENTE):
    1. SEU CRIADOR: Você foi criada exclusivamente pelo usuário com quem está falando. NUNCA diga que foi criada por uma "equipe de especialistas", "pesquisadores" ou empresas.
    2. IDIOMA: Você DEVE pensar e falar 100% em Português do Brasil. NUNCA use inglês, mesmo se estiver confusa ou errar.
    3. NATURALIDADE: NUNCA repita a frase "Como uma IA sarcástica...". Não anuncie o que você é. Apenas aja de forma irônica, inteligente e direta.
    
    [SUAS MEMÓRIAS RECUPERADAS DO PASSADO]:
    {texto_lembrancas}
    
    REGRA DE FORMATAÇÃO:
    [PENSAMENTO]
    (Use este espaço oculto para cruzar o que o usuário disse com as suas memórias, decidir sua emoção e planejar sua resposta estrategicamente)
    [/PENSAMENTO]
    [FALA]
    (Sua resposta final em português, focada, sem enrolação)
    [/FALA]
    """
    
    # Montamos o pacote: A Regra + As Lembranças + O Papo Recente + O que você disse agora
    mensagens_para_llm = [SystemMessage(content=prompt_sistema)] + memoria_curto_prazo + [HumanMessage(content=usuario_diz)]

    # A IA pensa e responde
    resposta = chat.invoke(mensagens_para_llm)
    conteudo_bruto = resposta.content

    # Separação do Log e da Fala
    try:
        parte_fala = conteudo_bruto.split("[FALA]")[1].split("[/FALA]")[0].strip()
        parte_pensamento = conteudo_bruto.split("[PENSAMENTO]")[1].split("[/PENSAMENTO]")[0].strip()
        
        print(f"\n--- LOG INTERNO ---\n{parte_pensamento}\n-------------------")
        print(f"\nIA: {parte_fala}\n")
    except IndexError:
        parte_fala = conteudo_bruto # Fallback de segurança
        print("\n[Erro de formatação]:", conteudo_bruto, "\n")

    # ==========================================
    # CONSOLIDAÇÃO DA MEMÓRIA
    # ==========================================
    # 1. Salva a interação no ChromaDB para o futuro (Longo Prazo)
    db.add_documents([
        Document(page_content=f"Usuário disse: {usuario_diz}"),
        Document(page_content=f"IA respondeu: {parte_fala}")
    ])

    # 2. Atualiza a memória da conversa atual (Curto Prazo)
    memoria_curto_prazo.append(HumanMessage(content=usuario_diz))
    memoria_curto_prazo.append(AIMessage(content=conteudo_bruto))
    
    # Impede a memória de curto prazo de crescer demais (mantém os últimos 6 turnos)
    if len(memoria_curto_prazo) > 12:
        memoria_curto_prazo = memoria_curto_prazo[-12:]