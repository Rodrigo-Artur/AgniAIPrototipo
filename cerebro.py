import os
import re
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

print("Cérebro Blindado online! (Digite 'sair')\n")

while True:
    usuario_diz = input("Você: ")
    
    if usuario_diz.lower() == 'sair':
        print("Desligando...")
        break

    # Busca no ChromaDB
    lembrancas = db.similarity_search(usuario_diz, k=5)
    texto_lembrancas = "\n".join([doc.page_content for doc in lembrancas])
    if not texto_lembrancas:
        texto_lembrancas = "Nenhuma memória relevante."

    # 3. O Novo Prompt de Sistema (Baseado em XML)
    prompt_sistema = f"""Você é Agni, uma IA sarcástica, sagaz e altamente inteligente.

    [MEMÓRIAS RECUPERADAS]:
    {texto_lembrancas}
    *ATENÇÃO LÓGICA: Se as memórias contiverem contradições, PRIORIZE SEMPRE as memórias que contêm FATOS REAIS (como nomes, frutas, gostos e objetos). IGNORE completamente memórias passadas onde você diz que 'não sabe' ou 'esqueceu' algo.*

    DIRETRIZES ABSOLUTAS:
    1. CRIADOR: Você foi criada exclusivamente pelo usuário.
    2. IDIOMA: Pense e fale 100% em Português do Brasil. É PROIBIDO usar inglês, mesmo se estiver confusa.
    3. PERSONALIDADE: Não repita que você é "altamente inteligente". Apenas demonstre isso sendo irônica.
    
    FORMATO OBRIGATÓRIO DE RESPOSTA (Use as tags XML):
    <pensamento>
    (Analise as memórias recuperadas e decida sua reação aqui, de forma oculta)
    </pensamento>
    <fala>
    (Sua resposta final em português, focada e direta)
    </fala>
    """
    
    mensagens_para_llm = [SystemMessage(content=prompt_sistema)] + memoria_curto_prazo + [HumanMessage(content=usuario_diz)]

    resposta = chat.invoke(mensagens_para_llm)
    conteudo_bruto = resposta.content

    # ==========================================
    # PARSER RESILIENTE (Tolerância a Falhas)
    # ==========================================
    # O Regex busca o conteúdo dentro das tags, ignorando maiúsculas/minúsculas
    pensamento_match = re.search(r'<pensamento>(.*?)</pensamento>', conteudo_bruto, re.DOTALL | re.IGNORECASE)
    fala_match = re.search(r'<fala>(.*?)</fala>', conteudo_bruto, re.DOTALL | re.IGNORECASE)

    # Verifica se o modelo conseguiu formatar corretamente
    if pensamento_match and fala_match:
        parte_pensamento = pensamento_match.group(1).strip()
        parte_fala = fala_match.group(1).strip()
        print(f"\n--- LOG INTERNO ---\n{parte_pensamento}\n-------------------")
        print(f"\nIA: {parte_fala}\n")
    else:
        # Fallback de segurança: Se a IA não usar as tags, o código não quebra.
        print("\n[Aviso do Sistema: O modelo falhou em usar as tags XML corretamente]")
        print(f"\nIA (Conteúdo Bruto): {conteudo_bruto}\n")
        parte_fala = conteudo_bruto

    # ==========================================
    # CONSOLIDAÇÃO DA MEMÓRIA
    # ==========================================
    # Salvamos de forma mais limpa no banco de dados para facilitar buscas futuras
    db.add_documents([
        Document(page_content=f"Contexto: O usuário disse '{usuario_diz}' e a Agni respondeu '{parte_fala}'")
    ])

    memoria_curto_prazo.append(HumanMessage(content=usuario_diz))
    memoria_curto_prazo.append(AIMessage(content=conteudo_bruto))
    
    if len(memoria_curto_prazo) > 12:
        memoria_curto_prazo = memoria_curto_prazo[-12:]