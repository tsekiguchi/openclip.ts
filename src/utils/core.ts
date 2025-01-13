
/**
 * @file Core utility functions/classes for Transformers.js.
 *
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 *
 * @module utils/core
 */

/**
 * Represents the initial progress information.
 */
export interface InitiateProgressInfo {
  /** The status of the progress. */
  status: 'initiate';

  /** The model ID or directory path. */
  name: string;

  /** The name of the file. */
  file: string;
}

/**
 * Represents the download progress information.
 */
export interface DownloadProgressInfo {
  /** The status of the progress. */
  status: 'download';

  /** The model ID or directory path. */
  name: string;

  /** The name of the file. */
  file: string;
}

/**
 * Represents the progress status information.
 */
export interface ProgressStatusInfo {
  /** The status of the progress. */
  status: 'progress';

  /** The model ID or directory path. */
  name: string;

  /** The name of the file. */
  file: string;

  /** A number between 0 and 100 representing the progress percentage. */
  progress: number;

  /** The number of bytes loaded. */
  loaded: number;

  /** The total number of bytes to be loaded. */
  total: number;
}

/**
 * Represents the completed progress information.
 */
export interface DoneProgressInfo {
  /** The status of the progress. */
  status: 'done';

  /** The model ID or directory path. */
  name: string;

  /** The name of the file. */
  file: string;
}

/**
 * Represents the readiness progress information.
 */
export interface ReadyProgressInfo {
  /** The status of the progress. */
  status: 'ready';

  /** The loaded task. */
  task: string;

  /** The loaded model. */
  model: string;
}

/**
 * Represents the union of all progress information types.
 */
export type ProgressInfo =
  | InitiateProgressInfo
  | DownloadProgressInfo
  | ProgressStatusInfo
  | DoneProgressInfo
  | ReadyProgressInfo;

/**
 * A callback function that is called with progress information.
 */
export type ProgressCallback = (progressInfo: ProgressInfo) => void;


/**
 * Helper function to dispatch progress callbacks.
 *
 * @param progress_callback The progress callback function to dispatch.
 * @param data The data to pass to the progress callback function.
 * @returns
 */
export function dispatchCallback(progress_callback?: ProgressCallback, data?: ProgressInfo): void {
	if (progress_callback && data) progress_callback(data);
}

/**
* Reverses the keys and values of an object.
*
* @param {Object} data The object to reverse.
* @returns {Object} The reversed object.
* @see https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
*/
export function reverseDictionary(data: object): object {
	// https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
	return Object.fromEntries(Object.entries(data).map(([key, value]) => [value, key]));
}

/**
* Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
*
* @param {string} string The string to escape.
* @returns {string} The escaped string.
*/
export function escapeRegExp(string: string): string {
	return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
* Check if a value is a typed array.
* @param val The value to check.
* @returns True if the value is a `TypedArray`, false otherwise.
*
* Adapted from https://stackoverflow.com/a/71091338/13989043
*/
export function isTypedArray(val: any): boolean {
	return val?.prototype?.__proto__?.constructor?.name === 'TypedArray';
}


/**
* Check if a value is an integer.
* @param x The value to check.
* @returns True if the value is a string, false otherwise.
*/
export function isIntegralNumber(x: any): boolean {
	return Number.isInteger(x) || typeof x === 'bigint'
}

/**
* Determine if a provided width or height is nullish.
* @param x The value to check.
* @returns True if the value is `null`, `undefined` or `-1`, false otherwise.
*/
export function isNullishDimension(x: any): boolean {
	return x === null || x === undefined || x === -1;
}

/**
* Calculates the dimensions of a nested array.
*
* @param arr The nested array to calculate dimensions for.
* @returns An array containing the dimensions of the input array.
*/
export function calculateDimensions(arr: any[]): number[] {
	const dimensions = [];
	let current = arr;
	while (Array.isArray(current)) {
			dimensions.push(current.length);
			current = current[0];
	}
	return dimensions;
}

/**
* Replicate python's .pop() method for objects.
* @param obj The object to pop from.
* @param key The key to pop.
* @param defaultValue The default value to return if the key does not exist.
* @returns The value of the popped key.
* @throws If the key does not exist and no default value is provided.
*/
export function pop<T extends object, K extends keyof T>(
  obj: T,
  key: K,
  defaultValue?: T[K]
): T[K] {
  const value = obj[key];
  if (value !== undefined) {
    delete obj[key];
    return value;
  }
  if (defaultValue === undefined) {
    throw new Error(`Key ${String(key)} does not exist in object.`);
  }
  return defaultValue;
}


/**
* Efficiently merge arrays, creating a new copy.
* Adapted from https://stackoverflow.com/a/6768642/13989043
* @param arrs Arrays to merge.
* @returns The merged array.
*/
export function mergeArrays(...arrs: Array<any>[]): Array<any> {
	return Array.prototype.concat.apply([], arrs);
}

/**
* Compute the Cartesian product of given arrays
* @param a Arrays to compute the product
* @returns Returns the computed Cartesian product as an array
* @see https://stackoverflow.com/a/43053803
*/
export function product(...a: Array<any>[]): Array<any> {
	return a.reduce((a, b) => a.flatMap(d => b.map(e => [d, e])));
}

/**
* Calculates the index offset for a given index and window size.
* @param i The index.
* @param w The window size.
* @returns The index offset.
*/
export function calculateReflectOffset(i: number, w: number): number {
	return Math.abs((i + w) % (2 * w) - w);
}

/**
 * Picks specified properties from an object.
 *
 * @param o The source object.
 * @param props The keys to pick from the object.
 * @returns A new object containing only the picked properties.
 */
export function pick<T extends object, K extends keyof T>(
  o: T,
  props: K[]
): Pick<T, K> {
  return Object.assign(
    {},
    ...props
      .map((prop) => 
        prop in o ? { [prop]: o[prop] } : null // Only return valid properties
      )
      .filter(Boolean) // Remove null values
  );
}



/**
* Calculate the length of a string, taking multi-byte characters into account.
* This mimics the behavior of Python's `len` function.
* @param s The string to calculate the length of.
* @returns The length of the string.
*/
export function len(s: string): number {
	let length = 0;
	for (const c of s) ++length;
	return length;
}

/**
* Count the occurrences of a value in an array or string.
* This mimics the behavior of Python's `count` method.
* @param arr The array or string to search.
* @param value The value to count.
*/
export function count(arr: any[] | string, value: any) {
	let count = 0;
	for (const v of arr) {
			if (v === value) ++count;
	}
	return count;
}
