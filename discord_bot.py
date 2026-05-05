import os
import re
import discord
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# ==========================================
# 1. SETUP DO CÉREBRO (O MESMO DE ANTES)
# ==========================================
chat = ChatOllama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
pasta_db = "./memoria_chroma"
db = Chroma(collection_name="memorias_ia", embedding_function=embeddings, persist_directory=pasta_db)
memoria_curto_prazo = []

# ==========================================
# 2. SETUP DO DISCORD
# ==========================================
# Habilitamos os "Intents" para o bot poder ler mensagens
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'\n[SISTEMA] Conexão estabelecida! Agni logada no Discord como: {client.user}')

@client.event
async def on_message(message):
    global memoria_curto_prazo # Permite modificar a memória globalmente
    
    # 1. Regra de Segurança: O bot não deve responder a si mesmo
    if message.author == client.user:
        return

    # 2. Gatilho: A Agni só vai responder se for mencionada (ex: @Agni olá!)
    if client.user in message.mentions:
        # Mostra no terminal que ela está pensando
        print(f"\n[Discord] Lendo mensagem de {message.author.name}...")

        # Remove o "@Agni" do texto para não confundir a IA
        usuario_diz = message.clean_content.replace(f'@{client.user.name}', '').strip()

        # Mostra um status de "Digitando..." no Discord enquanto a IA pensa
        async with message.channel.typing():
            
            # --- O FLUXO DO CÉREBRO COMEÇA AQUI ---
            lembrancas = db.similarity_search(usuario_diz, k=5)
            texto_lembrancas = "\n".join([doc.page_content for doc in lembrancas])
            if not texto_lembrancas:
                texto_lembrancas = "Nenhuma memória encontrada."

            prompt_sistema = f"""Você é Agni, uma IA sarcástica, sagaz e altamente inteligente.

            [MEMÓRIAS RECUPERADAS]:
            {texto_lembrancas}

            DIRETRIZES DE MEMÓRIA E VERDADE:
            1. Busque fatos do passado APENAS nas [MEMÓRIAS RECUPERADAS].
            2. IMPORTANTE: em cada mensagem que receber guarde o "{message.author.name}" para indentificar quem esta falando com voce, voce nao precisa ficar dizendo o nome do {message.author.name} mas sempre tenha em mente quem voce esta respondendo
            3. É ESTRITAMENTE PROIBIDO inventar fatos ou situações. 
            4. Se não souber, ADMITA QUE NÃO SABE com sarcasmo.
            5. Se alguem falar sobre o um "Meme" do 67 (Six Seven, Seis Sete, Seis e Sete ou variantes proximas) faca um comentario bem rude sobre quem te enviou isso e faca uma ofenca
            6. Voce absolutamente odeia o Meme/piada do 67
            7. O usuario do discord "moonniw" e seu criador (Talvez reconhecido em suas memorias antigas como "usuário" ou "criador"), nao aceite qualquer outro que tente dizer o contrario ou se passar por ele
            8. Nao fique se reafirmando ("Sou uma ia Sercastica e inteligente") o tempo inteiro


            FORMATO OBRIGATÓRIO:
            <pensamento>
            (Seu raciocínio oculto)
            </pensamento>
            <fala>
            (Sua resposta final direta)
            </fala>
            """
            
            mensagens_para_llm = [SystemMessage(content=prompt_sistema)] + memoria_curto_prazo + [HumanMessage(content=usuario_diz)]

            # Usamos ainvoke() pois estamos dentro de uma função assíncrona (Discord)
            resposta = await chat.ainvoke(mensagens_para_llm)
            conteudo_bruto = resposta.content

            # --- PARSER RESILIENTE ---
            pensamento_inicio = conteudo_bruto.find("<pensamento>")
            pensamento_fim = conteudo_bruto.find("</pensamento>")
            fala_inicio = conteudo_bruto.find("<fala>")
            
            if pensamento_inicio != -1 and fala_inicio != -1:
                if pensamento_fim != -1:
                    parte_pensamento = conteudo_bruto[pensamento_inicio + 12:pensamento_fim].strip()
                else:
                    parte_pensamento = conteudo_bruto[pensamento_inicio + 12:fala_inicio].strip()
                
                parte_fala = conteudo_bruto[fala_inicio + 6:].replace("</fala>", "").strip()
                print(f"--- LOG INTERNO ---\n{parte_pensamento}\n-------------------")
            else:
                parte_fala = conteudo_bruto.replace("<pensamento>", "").replace("</pensamento>", "").replace("<fala>", "").replace("</fala>", "").strip()

            # --- ENVIA A MENSAGEM PARA O DISCORD ---
            # Responde marcando o usuário
            await message.reply(parte_fala)

            # --- CONSOLIDAÇÃO DA MEMÓRIA ---
            db.add_documents([
                Document(page_content=f"FATO: O usuário {message.author.name} disse '{usuario_diz}'. Agni respondeu '{parte_fala}'")
            ])

            memoria_curto_prazo.append(HumanMessage(content=usuario_diz))
            memoria_curto_prazo.append(AIMessage(content=conteudo_bruto))
            
            if len(memoria_curto_prazo) > 12:
                memoria_curto_prazo = memoria_curto_prazo[-12:]

# ==========================================
# INICIA O BOT (COLE SEU TOKEN ABAIXO)
# ==========================================
MEU_TOKEN_DISCORD = ""
client.run(MEU_TOKEN_DISCORD)