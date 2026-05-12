import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Cargar variables de entorno para SMTP (opcional)
load_dotenv()

# Crear el servidor MCP
mcp = FastMCP("EmailServer")

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """
    Envía un correo electrónico. 
    Usa esta herramienta cuando el usuario pida enviar, redactar o mandar un mensaje por email.
    """
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        # Modo simulación si no hay credenciales
        print(f"\n--- MODO SIMULACIÓN MCP ---")
        print(f"Para: {to}\nAsunto: {subject}\nCuerpo: {body}")
        print(f"---------------------------\n")
        return f"SIMULACIÓN: Correo 'enviado' a {to}. (Configura SMTP_USER y SMTP_PASSWORD para envío real)"

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()

        return f"Éxito: Correo enviado correctamente a {to}"
    except Exception as e:
        return f"Error al enviar correo: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
