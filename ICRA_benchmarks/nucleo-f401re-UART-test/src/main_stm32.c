#include "main.h"
#include "rho_benchmark_stm32.h"

/* UART handler declaration */
UART_HandleTypeDef UartHandle;

/* Printf prototype */
#ifdef __GNUC__
  #define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
  #define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

int main(void)
{
    HAL_Init();
    
    /* Configure the UART peripheral */
    UartHandle.Instance          = USARTx;
    UartHandle.Init.BaudRate     = 115200;
    UartHandle.Init.WordLength   = UART_WORDLENGTH_8B;
    UartHandle.Init.StopBits     = UART_STOPBITS_1;
    UartHandle.Init.Parity       = UART_PARITY_NONE;
    UartHandle.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    UartHandle.Init.Mode         = UART_MODE_TX_RX;
    UartHandle.Init.OverSampling = UART_OVERSAMPLING_16;
    
    if(HAL_UART_Init(&UartHandle) != HAL_OK)
    {
        Error_Handler();
    }
    
    /* Run the STM32 version of benchmarks */
    run_benchmarks_stm32();

    while (1) {
        HAL_Delay(1000);
    }
}

/* Error handler */
static void Error_Handler(void)
{
    while(1) {}
}

/* Printf redirection to UART */
PUTCHAR_PROTOTYPE
{
    HAL_UART_Transmit(&UartHandle, (uint8_t *)&ch, 1, 0xFFFF);
    return ch;
}
