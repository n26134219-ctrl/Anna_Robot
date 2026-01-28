
python -c "
import pyaudio

audio = pyaudio.PyAudio()

print('当前所有输入设备:')
print('=' * 60)

for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        marker = ' ← 你的麦克风?' if i == 11 else ''
        print(f'索引 {i}: {info[\"name\"]}{marker}')
        print(f'  输入通道: {info[\"maxInputChannels\"]}')
        print(f'  采样率: {int(info[\"defaultSampleRate\"])}')
        print('-' * 60)

print()
print('检查设备 11:')
try:
    info = audio.get_device_info_by_index(11)
    print(f'✅ 设备 11 存在: {info[\"name\"]}')
    print(f'   最大输入通道: {info[\"maxInputChannels\"]}')
    
    if info['maxInputChannels'] == 1:
        print('\\n⚠️  昨天是 2 通道，今天变成 1 通道了！')
        print('   解决方法: 改 DEVICE_CHANNELS = 1')
    elif info['maxInputChannels'] == 0:
        print('\\n❌ 设备 11 现在不是输入设备了')
        print('   解决方法: 改 DEVICE_INDEX 到正确的索引')
        
except Exception as e:
    print(f'❌ 设备 11 不存在: {e}')
    print('   你的麦克风索引可能变了！')

audio.terminate()
"