type LogLevel = "DEBUG" | "INFO" | "WARN" | "ERROR" | "NONE";

const LogLevels: Record<LogLevel, number> = {
	DEBUG: 0,
	INFO: 1,
	WARN: 2,
	ERROR: 3,
	NONE: 4,
};

class Logger {
	private currentLevel: number;

	constructor(level: LogLevel = "INFO") {
		this.currentLevel = LogLevels[level];
	}

	setLevel(level: LogLevel): void {
		this.currentLevel = LogLevels[level];
	}

	log(level: LogLevel, message: string, ...args: any[]): void {
		if (LogLevels[level] >= this.currentLevel) {
			console.log(`[${level}] ${message}`, ...args);
		}
	}

	debug(message: string, ...args: any[]): void {
		this.log("DEBUG", message, ...args);
	}

	info(message: string, ...args: any[]): void {
		this.log("INFO", message, ...args);
	}

	warn(message: string, ...args: any[]): void {
		this.log("WARN", message, ...args);
	}

	error(message: string, ...args: any[]): void {
		this.log("ERROR", message, ...args);
	}
}

export const logger = new Logger();
