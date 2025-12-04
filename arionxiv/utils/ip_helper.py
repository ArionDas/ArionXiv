"""
IP Detection Helper for MongoDB Atlas Connection Issues
"""

import requests
import logging

logger = logging.getLogger(__name__)


def get_public_ip():
    """Get current public IP address"""
    try:
        response = requests.get('https://api.ipify.org?format=json', timeout=5)
        ip = response.json().get('ip')
        return ip
    except Exception as e:
        logger.warning(f"Failed to get public IP via ipify: {e}")
        
    # Fallback methods
    fallback_services = [
        'https://icanhazip.com',
        'https://ident.me',
        'https://ipecho.net/plain'
    ]
    
    for service in fallback_services:
        try:
            response = requests.get(service, timeout=5)
            ip = response.text.strip()
            if ip:
                return ip
        except Exception:
            continue
    
    return None


def display_ip_whitelist_help(current_ip=None):
    """Display helpful message about IP whitelisting"""
    if current_ip is None:
        current_ip = get_public_ip()
    
    message = "\n" + "="*70 + "\n"
    message += "MongoDB Atlas IP Whitelisting Issue Detected\n"
    message += "="*70 + "\n"
    
    if current_ip:
        message += f"\nYour current public IP: {current_ip}\n"
    else:
        message += "\nCould not detect your current IP automatically.\n"
    
    message += "\nSolutions:\n"
    message += "\n1. FOR DEVELOPMENT (Recommended):"
    message += "\n   - Go to MongoDB Atlas Dashboard"
    message += "\n   - Network Access > IP Access List"
    message += "\n   - Click 'Add IP Address'"
    message += "\n   - Select 'Allow Access from Anywhere' (0.0.0.0/0)"
    message += "\n   - This allows all IPs and solves your issue permanently for dev"
    
    message += "\n\n2. FOR PRODUCTION (More Secure):"
    if current_ip:
        message += f"\n   - Add your current IP: {current_ip}"
    message += "\n   - Use a static IP or VPN"
    message += "\n   - Configure IP ranges for your infrastructure"
    
    message += "\n\n3. ALTERNATIVE: Use MongoDB Connection String with srv+mongodb://"
    message += "\n   - Some ISPs work better with direct connections"
    
    message += "\n\nMongoDB Atlas Dashboard:"
    message += "\n   https://cloud.mongodb.com/"
    
    message += "\n\n" + "="*70 + "\n"
    
    logger.warning(message)


def check_mongodb_connection_error(error_message):
    """Check if error is related to IP whitelisting and provide help"""
    ip_whitelist_indicators = [
        "not authorized",
        "IP address is not whitelisted",
        "connection refused",
        "timed out",
        "network access",
        "authentication failed"
    ]
    
    error_lower = str(error_message).lower()
    
    for indicator in ip_whitelist_indicators:
        if indicator in error_lower:
            display_ip_whitelist_help()
            return True
    
    return False
