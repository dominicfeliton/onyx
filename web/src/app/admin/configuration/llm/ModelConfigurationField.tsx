"use client";

import { ArrayHelpers, FieldArray, FormikProps, useField } from "formik";
import { ModelConfiguration } from "./interfaces";
import {
  BooleanFormField,
  ManualErrorMessage,
  SubLabel,
  TextFormField,
} from "@/components/Field";
import { Label } from "@/components/ui/label";
import { useEffect, useState } from "react";
import CreateButton from "@/refresh-components/buttons/CreateButton";
import IconButton from "@/refresh-components/buttons/IconButton";
import SvgX from "@/icons/x";
import SvgChevronRight from "@/icons/chevron-right";
import { cn } from "@/lib/utils";

function ModelConfigurationRow({
  name,
  index,
  arrayHelpers,
  formikProps,
  setError,
  isAdvancedOpen,
  onToggleAdvanced,
}: {
  name: string;
  index: number;
  arrayHelpers: ArrayHelpers;
  formikProps: FormikProps<{ model_configurations: ModelConfiguration[] }>;
  setError: (value: string | null) => void;
  isAdvancedOpen: boolean;
  onToggleAdvanced: () => void;
}) {
  const [, input] = useField(`${name}[${index}]`);
  useEffect(() => {
    if (!input.touched) return;
    setError((input.error as { name: string } | undefined)?.name ?? null);
  }, [input.touched, input.error]);

  return (
    <div key={index} className="flex flex-col w-full gap-2">
      <div className="flex flex-row w-full gap-4">
        <div
          className={`flex flex-[2] ${
            input.touched && input.error
              ? "border-2 border-error rounded-lg"
              : ""
          }`}
        >
          <TextFormField
            name={`${name}[${index}].name`}
            label=""
            placeholder={`model-name-${index + 1}`}
            removeLabel
            hideError
          />
        </div>
        <div className="flex flex-[1]">
          <TextFormField
            name={`${name}[${index}].max_input_tokens`}
            label=""
            placeholder="Default"
            removeLabel
            hideError
            type="number"
            min={1}
          />
        </div>
        <div className="flex items-center gap-1 w-28 justify-end">
          <button
            type="button"
            onClick={onToggleAdvanced}
            className={cn(
              "flex items-center text-xs px-2 py-1.5 rounded-lg transition-colors",
              "text-text-500 hover:text-text-700 hover:bg-background-100",
              isAdvancedOpen && "bg-background-100 text-text-700"
            )}
          >
            <SvgChevronRight
              className={cn(
                "w-4 h-4 transition-transform",
                isAdvancedOpen && "rotate-90"
              )}
            />
            <span className="ml-1">Advanced</span>
          </button>
          <IconButton
            disabled={formikProps.values.model_configurations.length <= 1}
            onClick={() => {
              if (formikProps.values.model_configurations.length > 1) {
                setError(null);
                arrayHelpers.remove(index);
              }
            }}
            icon={SvgX}
            secondary
          />
        </div>
      </div>
      {isAdvancedOpen && (
        <div className="ml-4 pl-4 border-l-2 border-border-200 py-2">
          <BooleanFormField
            name={`${name}[${index}].use_non_tool_calling_fast`}
            label="Use fast pipeline"
            subtext="Enable for models that should use tools quickly or don't support native function calling. Uses programmatic tool execution, does not support multi-turn tool calling."
            small
          />
        </div>
      )}
    </div>
  );
}

export function ModelConfigurationField({
  name,
  formikProps,
}: {
  name: string;
  formikProps: FormikProps<{ model_configurations: ModelConfiguration[] }>;
}) {
  const [errorMap, setErrorMap] = useState<{ [index: number]: string }>({});
  const [finalError, setFinalError] = useState<string | undefined>();
  const [advancedOpenMap, setAdvancedOpenMap] = useState<{
    [index: number]: boolean;
  }>({});

  const toggleAdvanced = (index: number) => {
    setAdvancedOpenMap((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <div className="pb-5 flex flex-col w-full">
      <div className="flex flex-col">
        <Label className="text-md">Model Configurations</Label>
        <SubLabel>
          Add models and customize the number of input tokens that they accept.
        </SubLabel>
      </div>
      <FieldArray
        name={name}
        render={(arrayHelpers: ArrayHelpers) => (
          <div className="flex flex-col">
            <div className="flex flex-col gap-4 py-4">
              <div className="flex">
                <Label className="flex flex-[2]">Model Name</Label>
                <Label className="flex flex-[1]">Max Input Tokens</Label>
                <div className="w-28" />
              </div>
              {formikProps.values.model_configurations.map((_, index) => (
                <ModelConfigurationRow
                  key={index}
                  name={name}
                  formikProps={formikProps}
                  arrayHelpers={arrayHelpers}
                  index={index}
                  isAdvancedOpen={advancedOpenMap[index] || false}
                  onToggleAdvanced={() => toggleAdvanced(index)}
                  setError={(message: string | null) => {
                    const newErrors = { ...errorMap };
                    if (message) {
                      newErrors[index] = message;
                    } else {
                      delete newErrors[index];
                      for (const key in newErrors) {
                        const numKey = Number(key);
                        if (numKey > index) {
                          const errorValue = newErrors[key];
                          if (errorValue !== undefined) {
                            // Ensure the value is not undefined
                            newErrors[numKey - 1] = errorValue;
                            delete newErrors[numKey];
                          }
                        }
                      }
                    }
                    setErrorMap(newErrors);
                    setFinalError(
                      Object.values(newErrors).filter((item) => item)[0]
                    );
                  }}
                />
              ))}
            </div>
            {finalError && (
              <ManualErrorMessage>{finalError}</ManualErrorMessage>
            )}
            <div>
              <CreateButton
                onClick={() => {
                  arrayHelpers.push({
                    name: "",
                    is_visible: true,
                    // Use null so Yup.number().nullable() accepts empty inputs
                    max_input_tokens: null,
                    use_non_tool_calling_fast: false,
                  });
                }}
                className="mt-3"
                type="button"
              >
                Add New
              </CreateButton>
            </div>
          </div>
        )}
      />
    </div>
  );
}
