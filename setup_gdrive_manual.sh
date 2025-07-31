#!/bin/bash

# Manual Google Drive Setup Script for CARE2025 Biodreamer
# This script provides step-by-step instructions for configuring Google Drive

echo "=========================================="
echo "Google Drive Manual Setup Guide"
echo "=========================================="

# Function to display detailed manual setup steps
manual_setup() {
    echo "由于rclone版本较老，我们需要手动配置。请按以下步骤操作："
    echo ""
    echo "步骤 1: 打开新的终端窗口"
    echo "----------------------------------------"
    echo "在新终端中运行: rclone config"
    echo ""
    
    echo "步骤 2: 创建新的远程连接"
    echo "----------------------------------------"
    echo "当看到以下选项时："
    echo "n) New remote"
    echo "s) Set configuration password" 
    echo "q) Quit config"
    echo "n/s/q> "
    echo ""
    echo "请输入: n"
    echo ""
    
    echo "步骤 3: 设置远程连接名称"
    echo "----------------------------------------" 
    echo "当提示 'name>' 时，请输入: gdrive"
    echo ""
    
    echo "步骤 4: 选择存储类型"
    echo "----------------------------------------"
    echo "在存储类型列表中找到 Google Drive，通常是选项 15 或 16"
    echo "输入对应的数字（如: 15）"
    echo ""
    
    echo "步骤 5: 客户端配置"
    echo "----------------------------------------"
    echo "client_id> 直接按回车（留空）"
    echo "client_secret> 直接按回车（留空）"
    echo ""
    
    echo "步骤 6: 权限范围"
    echo "----------------------------------------"
    echo "选择权限范围，通常选择 1 (full access)"
    echo "输入: 1"
    echo ""
    
    echo "步骤 7: 其他设置"
    echo "----------------------------------------"
    echo "root_folder_id> 直接按回车（留空）"
    echo "service_account_file> 直接按回车（留空）"
    echo ""
    
    echo "步骤 8: 高级配置"
    echo "----------------------------------------"
    echo "当询问 'Edit advanced config?' 时"
    echo "输入: n"
    echo ""
    
    echo "步骤 9: 自动配置"
    echo "----------------------------------------"
    echo "当询问 'Use auto config?' 时"
    echo "输入: y"
    echo ""
    echo "这会打开浏览器进行Google账户认证"
    echo "请在浏览器中："
    echo "1. 登录你的Google账户"
    echo "2. 允许rclone访问Google Drive"
    echo "3. 复制授权码返回终端"
    echo ""
    
    echo "步骤 10: 团队盘设置"
    echo "----------------------------------------"
    echo "当询问 'Configure this as a team drive?' 时"
    echo "输入: n"
    echo ""
    
    echo "步骤 11: 确认配置"
    echo "----------------------------------------"
    echo "检查配置信息，如果正确，输入: y"
    echo ""
    
    echo "步骤 12: 退出配置"
    echo "----------------------------------------"
    echo "输入: q 退出配置程序"
    echo ""
    
    echo "配置完成后，运行以下命令测试："
    echo "rclone lsd gdrive:"
    echo ""
}

# Function to create a simple config file
create_simple_config() {
    echo "或者，我可以为你创建一个基础配置文件，然后你只需要获取授权码："
    echo ""
    
    CONFIG_DIR="$HOME/.config/rclone"
    CONFIG_FILE="$CONFIG_DIR/rclone.conf"
    
    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_DIR"
    
    # Create basic config
    cat > "$CONFIG_FILE" << 'EOF'
[gdrive]
type = drive
scope = drive
EOF
    
    echo "✅ 基础配置文件已创建: $CONFIG_FILE"
    echo ""
    echo "现在你需要获取访问令牌："
    echo "1. 访问: https://rclone.org/drive/#making-your-own-client-id"
    echo "2. 按照说明创建Google API凭据"
    echo "3. 运行: rclone config reconnect gdrive:"
    echo "4. 在浏览器中完成授权"
    echo ""
}

# Function to test existing config
test_config() {
    echo "测试现有配置..."
    
    if [ ! -f "$HOME/.config/rclone/rclone.conf" ]; then
        echo "❌ 配置文件不存在"
        return 1
    fi
    
    if rclone listremotes | grep -q "gdrive:"; then
        echo "✅ 找到 gdrive 远程连接"
        
        if rclone lsd gdrive: &> /dev/null; then
            echo "✅ Google Drive 连接成功！"
            echo ""
            echo "现在你可以运行上传脚本:"
            echo "./upload_to_gdrive.sh upload"
            return 0
        else
            echo "❌ Google Drive 连接失败，需要重新授权"
            echo "运行: rclone config reconnect gdrive:"
            return 1
        fi
    else
        echo "❌ 未找到 gdrive 远程连接"
        return 1
    fi
}

# Main menu
echo "请选择一个选项:"
echo "1) 显示详细手动配置步骤"
echo "2) 创建基础配置文件"  
echo "3) 测试现有配置"
echo "4) 退出"
echo ""

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        manual_setup
        ;;
    2)
        create_simple_config
        ;;
    3)
        test_config
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac 